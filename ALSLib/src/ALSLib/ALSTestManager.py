import sys, os, threading, boto3
from botocore.exceptions import ClientError
from datetime import datetime, timezone

from . import ALSClient

s3 = boto3.resource('s3')
LOCAL_TEST_ID = 'test_id_local'
LOCAL_TESTSUITE_ID = 'testsuite_id_local'
LOCAL_OVERRIDES = 'not defined'
LOCAL_CUSTOMER_ID = 'customer_id'

INSTANCES_SIMULATION_STATUS_READY = 'READY'
TESTS_STATUS_PENDING = 'PENDING'
TESTS_STATUS_FAILED = 'FAILED'
TESTS_STATUS_ERROR = 'ERROR'
TESTS_STATUS_RUNNING = 'RUNNING'
TESTS_STATUS_SUCCESS = 'SUCCESS'
RESULTS_FOLDER = 'Results'

CLIENT_RESPONSE_DONE = 'Done!'
CLIENT_RESPONSE_NO_SCENARIO = 'error No Scenario found with this name'


# Set default values for environment variables if not present
os.environ.setdefault('AWS_REGION', 'eu-central-1')
os.environ.setdefault('TEST_ID', LOCAL_TEST_ID)
os.environ.setdefault('INSTANCE_IP', '127.0.0.1')
os.environ.setdefault('RESULTS_DIR', '/tmp/results')
os.environ.setdefault('OVERRIDE', LOCAL_OVERRIDES)
os.environ.setdefault('S3_BUCKET_NAME', 'ailivesim-customer-shared-files')
os.environ.setdefault('DYNAMODB_TEST_TABLE_NAME', 'als-tests')

class ThreadUploadingFile(threading.Thread):
    def __init__(self, local_path, bucket_name, s3_path):
        """Initialize the thread"""
        threading.Thread.__init__(self)
        self.local_path = local_path
        self.bucket_name = bucket_name
        self.s3_path = s3_path

    def run(self):
        """Run the thread"""
        s3.meta.client.upload_file(self.local_path, self.bucket_name, self.s3_path)


def safe_print(*args):
    sep = " "
    joined_string =  "OK:"+ sep.join([ str(arg) for arg in args ])
    print(joined_string)
    # print(joined_string  + "\n", sep=sep, end=end, **kwargs)
    sys.stdout.flush()

class ALSTestManager:
    ALS_ERROR = -1
    ALS_SUCCESS = 0
    ALS_FAIL = 1

    lock = threading.Lock()
    testEnded = False
    echo_port = 9000

    def __init__(self, fctMessageHandler):
        if len(sys.argv) >= 2:
            safe_print(f'Running local test with scenario: {sys.argv[1]}')
            self.scenario = sys.argv[1]
            self.overrides = sys.argv[2] if len(sys.argv) == 3 else LOCAL_OVERRIDES
        else:
            self.scenario = os.getenv('SCENARIO', None)
            self.overrides = os.getenv('OVERRIDES', LOCAL_OVERRIDES)

        if not 'TEST_ID' in os.environ:
            safe_print('Error: TEST_ID is not defined in environment variables.')
            exit(1)

        if not 'INSTANCE_IP' in os.environ:
            safe_print('Error: INSTANCE_IP is not defined in environment variables.')
            exit(1)

        self.test_id = os.environ['TEST_ID']
        self.instance_ip = os.environ['INSTANCE_IP']
        self.is_local_run = self.test_id == LOCAL_TEST_ID
        self.aws_region = os.getenv('AWS_REGION', 'eu-central-1')
        self.s3_bucket_name = os.getenv('S3_BUCKET_NAME')
        self.dynamo_tests_db = os.getenv('DYNAMODB_TEST_TABLE_NAME')

        if not self.is_local_run:
            safe_print('Connecting to database...')
            self.dynamodb_client = boto3.client('dynamodb', region_name=self.aws_region)
            safe_print('Connected.')

        self.details = self.GetTestDetails()
        self.client = ALSClient.Client((self.details['InstanceIP'], self.echo_port), fctMessageHandler)

    def UploadToS3(self, filename, content):
        s3 = boto3.resource("s3")
        CustomerId, TestSuiteId, TestId = self.details["CustomerId"], self.details["TestSuiteId"], self.details["Id"]
        directory = "{0}/{1}/{2}/{3}".format(CustomerId, RESULTS_FOLDER, TestSuiteId, TestId)
        obj = s3.Object(self.s3_bucket_name, "{0}/{1}".format(directory, filename))
        obj.put(Body=content)

    def SaveReplayFiles(self):
        print("getting replays files..")
        try:
            replayFiles = self.client.request_get_latest_replay_files()
            self.client.wait_for_task_complete()
        except Exception as e:
            print("UNSPECIFIED ERROR TRYING TO GET REPLAYS")
            print(e)
            return
        if replayFiles is None:
            print("UNSPECIFIED ERROR TRYING TO GET REPLAYS")
            self.client.connect()
            return
        filename = replayFiles.split("<Replay>")[1].split(".txt</Replay>")[0]
        if self.is_local_run:
            safe_print("Store replay files locally.")
            f = open("{0}.xml".format(filename), "a")
            f.write(replayFiles.split('</Scenario>')[0] + '</Scenario>')
            f.close()
            f = open("{0}.txt".format(filename), "a")
            f.write(replayFiles.split('</Scenario>')[1])
            f.close()
        else:
            safe_print("Uploading replay files to S3...")
            self.UploadToS3("{0}.xml".format(filename), replayFiles.split('</Scenario>')[0] + '</Scenario>')
            self.UploadToS3("{0}.txt".format(filename), replayFiles.split('</Scenario>')[1])
            safe_print("Uploaded replay files.")

    def UpdateTestStatus(self, value, error_message):
        try:
            updated_values = self.dynamodb_client.update_item(
                ExpressionAttributeNames={
                    '#S': 'Status',
                    '#EM': 'ErrorMessage',
                    '#UA': 'UpdatedAt'
                },
                ExpressionAttributeValues={
                    ':s': {	'S': value },
                    ':em': { 'S': error_message },
                    ':ua': {'S': datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"}
                },
                Key={
                    'Id': { 'S': self.test_id }
                },
                ReturnValues='ALL_NEW',
                TableName=self.dynamo_tests_db,
                UpdateExpression='SET #S = :s, #EM = :em, #UA = :ua',
            )
        except ClientError as e:
            safe_print("Error: {0}".format(e.response['Error']['Message']))
            return False
        return updated_values['Attributes']['Status'] == value

    def SetTestStatusSuccess(self):
        if self.is_local_run:
            safe_print("Test succeed")
            return self.ALS_SUCCESS
        return self.UpdateTestStatus(TESTS_STATUS_SUCCESS, ' ')

    def SetTestStatusFailed(self):
        if self.is_local_run:
            safe_print("Test failed")
            return self.ALS_FAIL
        return self.UpdateTestStatus(TESTS_STATUS_FAILED, ' ')

    def SetTestStatusError(self, error_message):
        safe_print('ERROR ({0}): {1}'.format(self.test_id, error_message))
        safe_print('Test status updated to error')
        if self.is_local_run:
            return self.ALS_ERROR
        return self.UpdateTestStatus(TESTS_STATUS_ERROR, error_message)

    def SetTestStatusRunning(self):
        if self.is_local_run:
            safe_print("Test running.")
            return self.ALS_SUCCESS
        return self.UpdateTestStatus(TESTS_STATUS_RUNNING, ' ')

    def GetTestDetails(self):
        test_id = self.test_id
        if self.is_local_run:
            test = {
                'Id': test_id,
                'TestSuiteId': 'testsuite_id_local',
                'Status': TESTS_STATUS_PENDING,
                'Scenario': self.scenario,
                'Overrides': self.overrides,
                'InstanceIP': self.instance_ip,
                'CustomerId': 'localhost'
            }
            return test

        safe_print('Fetching test details...')
        try:
            test_raw_db = self.dynamodb_client.get_item(
                TableName=self.dynamo_tests_db,
                Key={
                    'Id': { 'S': test_id }
                }
            )

            if (not ('ResponseMetadata' in test_raw_db) or test_raw_db['ResponseMetadata']['HTTPStatusCode'] != 200):
                self.SetTestStatusError('Error: can\'t connect to the database.')
                exit(1)

            if not ('Item' in test_raw_db):
                self.SetTestStatusError("Error: {0} does not exist.".format(test_id))
                exit(1)

            safe_print(test_raw_db)
            #sys.stdout.flush()

            test = {
                'Id': test_raw_db['Item']['Id']['S'],
                'TestSuiteId': test_raw_db['Item']['TestSuiteId']['S'],
                'Status': test_raw_db['Item']['Status']['S'],
                'Scenario': test_raw_db['Item']['Scenario']['S'],
                'Overrides': test_raw_db['Item']['Overrides']['S'],
                'CustomerId': test_raw_db['Item']['CustomerId']['S'],
                'InstanceIP': self.instance_ip,
            }
        except ClientError as e:
            self.SetTestStatusError('Error: {0}'.format(e.response['Error']['Message']))
            exit(1)
        except KeyError as e:
            self.SetTestStatusError('Error: {0}'.format(e))
            exit(1)
        except:
            e = sys.exc_info()[0]
            self.SetTestStatusError('Error: {0}'.format(e))
            exit(1)

        safe_print('Test details successfully fetched.')
        #sys.stdout.flush()
        return test

    def StartScenario(self):
        safe_print("Connecting to instance...")
        timeout = int(os.getenv('AILS_TIMEOUT', 15))
        print(f"CONNECT TIMEOUT: {timeout}")
        #sys.stdout.flush()
        r = self.client.connect(timeout)
        if r == False:
            self.SetTestStatusError("Error: Can't connect to {0}".format(self.details["InstanceIP"]))
            return self.ALS_ERROR
        safe_print("Connected !")
        #sys.stdout.flush()

        Scenario, Overrides = self.details["Scenario"], self.details["Overrides"]
        safe_print("Loading scenario {0} (overrides: {1})".format(Scenario, Overrides))
        #sys.stdout.flush()

        print(f"LOAD TIMEOUT: {timeout}")

        if Overrides != "not defined":
            self.client.request_load_scenario_with_overrides(Scenario, Overrides, timeout=float(timeout))
        else:
            self.client.request_load_scenario(Scenario, timeout=float(timeout))
        self.client.wait_for_task_complete()

        if self.client.response != "Done!":
            if self.client.response == "error No Scenario found with this name":
                self.SetTestStatusError("Error: Scenario {0} does not exist".format(self.details["Scenario"]))
            else:
                self.SetTestStatusError(self.client.response or 'Error unknown')
            return self.ALS_ERROR
        return self.ALS_SUCCESS

    def StopScenario(self):
        self.client.request_toggle_pause()
        self.client.request_destroy_situation()
        self.client.request_toggle_pause()

    def StartTest(self, fct):
        self.SetTestStatusRunning()
        if self.StartScenario() == self.ALS_ERROR:
            return

        timeout = int(os.getenv('AILS_TIMEOUT', 15))

        self.client.connect(30)
        self.client.request_load_scenario_with_overrides(self.scenario, self.overrides, timeout=float(timeout))

        result = fct()
        if result == self.ALS_SUCCESS:
            self.SetTestStatusSuccess()
        elif result == self.ALS_FAIL:
            self.SetTestStatusFailed()

    def SaveSensorDataFilesToS3(self, local_directory):
        nb_files = 0
        threads = []

        for root, _, files in os.walk(local_directory):
            for filename in files:
                local_path = os.path.join(root, filename)
                CustomerId, TestSuiteId, TestId = self.details["CustomerId"], self.details["TestSuiteId"], self.details["Id"]
                directory = os.path.join(CustomerId, RESULTS_FOLDER, TestSuiteId, TestId)
                destination_path = os.path.join(directory, filename).replace('\\','/')

                if self.is_local_run:
                    safe_print("{0} would be uploaded to {1}- but now in local mode.".format(local_path, destination_path))
                    continue

                safe_print("Uploading {0} to {1}...".format(local_path, destination_path))

                my_thread = ThreadUploadingFile(local_path, self.s3_bucket_name, destination_path)
                my_thread.start()
                threads.append(my_thread)
                nb_files += 1

        safe_print("Waiting for threads...")
        for t in threads:
            t.join()
        safe_print("Uploaded {0} files.".format(nb_files))
