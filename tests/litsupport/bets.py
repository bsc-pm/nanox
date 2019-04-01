import os
import lit.formats
import tempfile
import shutil

class BetsTest(lit.formats.FileBasedTest):
    """Bets tests
    """
    def __init__(self, execute_external=False):
        self.execute_external = execute_external

    def getTestsInDirectory(self, testSuite, path_in_suite,
                            litConfig, localConfig):
        source_path = testSuite.getSourcePath(path_in_suite)
        for filename in os.listdir(source_path):
            # Ignore dot files and excluded tests.
            if (filename.startswith('.') or
                filename in localConfig.excludes):
                continue

            filepath = os.path.join(source_path, filename)
            if not os.path.isdir(filepath):
                if any(filename.endswith(suffix) for suffix in localConfig.suffixes):
                    yield lit.Test.Test(testSuite, path_in_suite + (filename,),
                                        localConfig)

    def execute(self, test, litConfig):
        sourcePath = test.getSourcePath()
        tempFile = tempfile.NamedTemporaryFile(mode="w+t")
        cmd = [test.config.bets, "-nocolor", "-o", tempFile.name, sourcePath]

        result_log = ""
        try:
            out, err, exitCode = lit.util.executeCommand(
                cmd, env=test.config.environment,
                timeout=litConfig.maxIndividualTestTime)
            tempFile.seek(0)
            for line in tempFile:
                result_log += "[BETS LOG] {}".format(line)
        except lit.util.ExecuteCommandTimeoutException:
            return (lit.Test.TIMEOUT,
                    'Reached timeout of {} seconds'.format(
                        litConfig.maxIndividualTestTime)
                   )
        finally:
            tempFile.close()

        test_result = lit.Test.FAIL if exitCode else lit.Test.PASS
        bets_cmd = "[BETS COMMAND]: {}\n".format(" ".join(cmd))

        return test_result, out + err + bets_cmd + result_log
