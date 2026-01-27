"""Unit test for the gtest_xml_output module"""
import datetime
import errno
import os
import re
import sys
from xml.dom import minidom, Node
from googletest.test import gtest_test_utils
from googletest.test import gtest_xml_test_utils
GTEST_FILTER_FLAG = '--gtest_filter'
GTEST_LIST_TESTS_FLAG = '--gtest_list_tests'
GTEST_OUTPUT_FLAG = '--gtest_output'
GTEST_DEFAULT_OUTPUT_FILE = 'test_detail.xml'
GTEST_PROGRAM_NAME = 'gtest_xml_output_unittest_'
NO_STACKTRACE_SUPPORT_FLAG = '--no_stacktrace_support'
TOTAL_SHARDS_ENV_VAR = 'GTEST_TOTAL_SHARDS'
SHARD_INDEX_ENV_VAR = 'GTEST_SHARD_INDEX'
SHARD_STATUS_FILE_ENV_VAR = 'GTEST_SHARD_STATUS_FILE'
SUPPORTS_STACK_TRACES = NO_STACKTRACE_SUPPORT_FLAG not in sys.argv
if SUPPORTS_STACK_TRACES:
    STACK_TRACE_TEMPLATE = '\nStack trace:\n*'
    STACK_TRACE_ENTITY_TEMPLATE = ''
else:
    STACK_TRACE_TEMPLATE = '\n'
    STACK_TRACE_ENTITY_TEMPLATE = '&#x0A;'
    sys.argv.remove(NO_STACKTRACE_SUPPORT_FLAG)
EXPECTED_NON_EMPTY_XML = '<?xml version="1.0" encoding="UTF-8"?>\n<testsuites tests="26" failures="5" disabled="2" errors="0" time="*" timestamp="*" name="AllTests" ad_hoc_property="42">\n  <testsuite name="SuccessfulTest" tests="1" failures="0" disabled="0" skipped="0" errors="0" time="*" timestamp="*">\n    <testcase name="Succeeds" file="gtest_xml_output_unittest_.cc" line="53" status="run" result="completed" time="*" timestamp="*" classname="SuccessfulTest"/>\n  </testsuite>\n  <testsuite name="FailedTest" tests="1" failures="1" disabled="0" skipped="0" errors="0" time="*" timestamp="*">\n    <testcase name="Fails" file="gtest_xml_output_unittest_.cc" line="61" status="run" result="completed" time="*" timestamp="*" classname="FailedTest">\n      <failure message="gtest_xml_output_unittest_.cc:*&#x0A;Expected equality of these values:&#x0A;  1&#x0A;  2%(stack_entity)s" type=""><![CDATA[gtest_xml_output_unittest_.cc:*\nExpected equality of these values:\n  1\n  2%(stack)s]]></failure>\n    </testcase>\n  </testsuite>\n  <testsuite name="MixedResultTest" tests="3" failures="1" disabled="1" skipped="0" errors="0" time="*" timestamp="*">\n    <testcase name="Succeeds" file="gtest_xml_output_unittest_.cc" line="88" status="run" result="completed" time="*" timestamp="*" classname="MixedResultTest"/>\n    <testcase name="Fails" file="gtest_xml_output_unittest_.cc" line="93" status="run" result="completed" time="*" timestamp="*" classname="MixedResultTest">\n      <failure message="gtest_xml_output_unittest_.cc:*&#x0A;Expected equality of these values:&#x0A;  1&#x0A;  2%(stack_entity)s" type=""><![CDATA[gtest_xml_output_unittest_.cc:*\nExpected equality of these values:\n  1\n  2%(stack)s]]></failure>\n      <failure message="gtest_xml_output_unittest_.cc:*&#x0A;Expected equality of these values:&#x0A;  2&#x0A;  3%(stack_entity)s" type=""><![CDATA[gtest_xml_output_unittest_.cc:*\nExpected equality of these values:\n  2\n  3%(stack)s]]></failure>\n    </testcase>\n    <testcase name="DISABLED_test" file="gtest_xml_output_unittest_.cc" line="98" status="notrun" result="suppressed" time="*" timestamp="*" classname="MixedResultTest"/>\n  </testsuite>\n  <testsuite name="XmlQuotingTest" tests="1" failures="1" disabled="0" skipped="0" errors="0" time="*" timestamp="*">\n    <testcase name="OutputsCData" file="gtest_xml_output_unittest_.cc" line="102" status="run" result="completed" time="*" timestamp="*" classname="XmlQuotingTest">\n      <failure message="gtest_xml_output_unittest_.cc:*&#x0A;Failed&#x0A;XML output: &lt;?xml encoding=&quot;utf-8&quot;&gt;&lt;top&gt;&lt;![CDATA[cdata text]]&gt;&lt;/top&gt;%(stack_entity)s" type=""><![CDATA[gtest_xml_output_unittest_.cc:*\nFailed\nXML output: <?xml encoding="utf-8"><top><![CDATA[cdata text]]>]]&gt;<![CDATA[</top>%(stack)s]]></failure>\n    </testcase>\n  </testsuite>\n  <testsuite name="InvalidCharactersTest" tests="1" failures="1" disabled="0" skipped="0" errors="0" time="*" timestamp="*">\n    <testcase name="InvalidCharactersInMessage" file="gtest_xml_output_unittest_.cc" line="109" status="run" result="completed" time="*" timestamp="*" classname="InvalidCharactersTest">\n      <failure message="gtest_xml_output_unittest_.cc:*&#x0A;Failed&#x0A;Invalid characters in brackets []%(stack_entity)s" type=""><![CDATA[gtest_xml_output_unittest_.cc:*\nFailed\nInvalid characters in brackets []%(stack)s]]></failure>\n    </testcase>\n  </testsuite>\n  <testsuite name="DisabledTest" tests="1" failures="0" disabled="1" skipped="0" errors="0" time="*" timestamp="*">\n    <testcase name="DISABLED_test_not_run" file="gtest_xml_output_unittest_.cc" line="68" status="notrun" result="suppressed" time="*" timestamp="*" classname="DisabledTest"/>\n  </testsuite>\n  <testsuite name="SkippedTest" tests="3" failures="1" disabled="0" skipped="2" errors="0" time="*" timestamp="*">\n    <testcase name="Skipped" status="run" file="gtest_xml_output_unittest_.cc" line="75" result="skipped" time="*" timestamp="*" classname="SkippedTest">\n      <skipped message="gtest_xml_output_unittest_.cc:*&#x0A;&#x0A;"><![CDATA[gtest_xml_output_unittest_.cc:*\n\n]]></skipped>\n    </testcase>\n    <testcase name="SkippedWithMessage" file="gtest_xml_output_unittest_.cc" line="79" status="run" result="skipped" time="*" timestamp="*" classname="SkippedTest">\n      <skipped message="gtest_xml_output_unittest_.cc:*&#x0A;It is good practice to tell why you skip a test.&#x0A;"><![CDATA[gtest_xml_output_unittest_.cc:*\nIt is good practice to tell why you skip a test.\n]]></skipped>\n    </testcase>\n    <testcase name="SkippedAfterFailure" file="gtest_xml_output_unittest_.cc" line="83" status="run" result="completed" time="*" timestamp="*" classname="SkippedTest">\n      <failure message="gtest_xml_output_unittest_.cc:*&#x0A;Expected equality of these values:&#x0A;  1&#x0A;  2%(stack_entity)s" type=""><![CDATA[gtest_xml_output_unittest_.cc:*\nExpected equality of these values:\n  1\n  2%(stack)s]]></failure>\n      <skipped message="gtest_xml_output_unittest_.cc:*&#x0A;It is good practice to tell why you skip a test.&#x0A;"><![CDATA[gtest_xml_output_unittest_.cc:*\nIt is good practice to tell why you skip a test.\n]]></skipped>\n    </testcase>\n\n  </testsuite>\n  <testsuite name="PropertyRecordingTest" tests="4" failures="0" disabled="0" skipped="0" errors="0" time="*" timestamp="*" SetUpTestSuite="yes" TearDownTestSuite="aye">\n    <testcase name="OneProperty" file="gtest_xml_output_unittest_.cc" line="121" status="run" result="completed" time="*" timestamp="*" classname="PropertyRecordingTest">\n      <properties>\n        <property name="key_1" value="1"/>\n      </properties>\n    </testcase>\n    <testcase name="IntValuedProperty" file="gtest_xml_output_unittest_.cc" line="125" status="run" result="completed" time="*" timestamp="*" classname="PropertyRecordingTest">\n      <properties>\n        <property name="key_int" value="1"/>\n      </properties>\n    </testcase>\n    <testcase name="ThreeProperties" file="gtest_xml_output_unittest_.cc" line="129" status="run" result="completed" time="*" timestamp="*" classname="PropertyRecordingTest">\n      <properties>\n        <property name="key_1" value="1"/>\n        <property name="key_2" value="2"/>\n        <property name="key_3" value="3"/>\n      </properties>\n    </testcase>\n    <testcase name="TwoValuesForOneKeyUsesLastValue" file="gtest_xml_output_unittest_.cc" line="135" status="run" result="completed" time="*" timestamp="*" classname="PropertyRecordingTest">\n      <properties>\n        <property name="key_1" value="2"/>\n      </properties>\n    </testcase>\n  </testsuite>\n  <testsuite name="NoFixtureTest" tests="3" failures="0" disabled="0" skipped="0" errors="0" time="*" timestamp="*">\n     <testcase name="RecordProperty" file="gtest_xml_output_unittest_.cc" line="140" status="run" result="completed" time="*" timestamp="*" classname="NoFixtureTest">\n       <properties>\n         <property name="key" value="1"/>\n       </properties>\n     </testcase>\n     <testcase name="ExternalUtilityThatCallsRecordIntValuedProperty" file="gtest_xml_output_unittest_.cc" line="153" status="run" result="completed" time="*" timestamp="*" classname="NoFixtureTest">\n       <properties>\n         <property name="key_for_utility_int" value="1"/>\n       </properties>\n     </testcase>\n     <testcase name="ExternalUtilityThatCallsRecordStringValuedProperty" file="gtest_xml_output_unittest_.cc" line="157" status="run" result="completed" time="*" timestamp="*" classname="NoFixtureTest">\n       <properties>\n         <property name="key_for_utility_string" value="1"/>\n       </properties>\n     </testcase>\n  </testsuite>\n  <testsuite name="Single/ValueParamTest" tests="4" failures="0" disabled="0" skipped="0" errors="0" time="*" timestamp="*">\n    <testcase name="HasValueParamAttribute/0" file="gtest_xml_output_unittest_.cc" line="164" value_param="33" status="run" result="completed" time="*" timestamp="*" classname="Single/ValueParamTest" />\n    <testcase name="HasValueParamAttribute/1" file="gtest_xml_output_unittest_.cc" line="164" value_param="42" status="run" result="completed" time="*" timestamp="*" classname="Single/ValueParamTest" />\n    <testcase name="AnotherTestThatHasValueParamAttribute/0" file="gtest_xml_output_unittest_.cc" line="165" value_param="33" status="run" result="completed" time="*" timestamp="*" classname="Single/ValueParamTest" />\n    <testcase name="AnotherTestThatHasValueParamAttribute/1" file="gtest_xml_output_unittest_.cc" line="165" value_param="42" status="run" result="completed" time="*" timestamp="*" classname="Single/ValueParamTest" />\n  </testsuite>\n  <testsuite name="TypedTest/0" tests="1" failures="0" disabled="0" skipped="0" errors="0" time="*" timestamp="*">\n    <testcase name="HasTypeParamAttribute" file="gtest_xml_output_unittest_.cc" line="173" type_param="*" status="run" result="completed" time="*" timestamp="*" classname="TypedTest/0" />\n  </testsuite>\n  <testsuite name="TypedTest/1" tests="1" failures="0" disabled="0" skipped="0" errors="0" time="*" timestamp="*">\n    <testcase name="HasTypeParamAttribute" file="gtest_xml_output_unittest_.cc" line="173" type_param="*" status="run" result="completed" time="*" timestamp="*" classname="TypedTest/1" />\n  </testsuite>\n  <testsuite name="Single/TypeParameterizedTestSuite/0" tests="1" failures="0" disabled="0" skipped="0" errors="0" time="*" timestamp="*">\n    <testcase name="HasTypeParamAttribute" file="gtest_xml_output_unittest_.cc" line="180" type_param="*" status="run" result="completed" time="*" timestamp="*" classname="Single/TypeParameterizedTestSuite/0" />\n  </testsuite>\n  <testsuite name="Single/TypeParameterizedTestSuite/1" tests="1" failures="0" disabled="0" skipped="0" errors="0" time="*" timestamp="*">\n    <testcase name="HasTypeParamAttribute" file="gtest_xml_output_unittest_.cc" line="180" type_param="*" status="run" result="completed" time="*" timestamp="*" classname="Single/TypeParameterizedTestSuite/1" />\n  </testsuite>\n</testsuites>' % {'stack': STACK_TRACE_TEMPLATE, 'stack_entity': STACK_TRACE_ENTITY_TEMPLATE}
EXPECTED_FILTERED_TEST_XML = '<?xml version="1.0" encoding="UTF-8"?>\n<testsuites tests="1" failures="0" disabled="0" errors="0" time="*"\n            timestamp="*" name="AllTests" ad_hoc_property="42">\n  <testsuite name="SuccessfulTest" tests="1" failures="0" disabled="0" skipped="0"\n             errors="0" time="*" timestamp="*">\n    <testcase name="Succeeds" file="gtest_xml_output_unittest_.cc" line="53" status="run" result="completed" time="*" timestamp="*" classname="SuccessfulTest"/>\n  </testsuite>\n</testsuites>'
EXPECTED_SHARDED_TEST_XML = '<?xml version="1.0" encoding="UTF-8"?>\n<testsuites tests="3" failures="0" disabled="0" errors="0" time="*" timestamp="*" name="AllTests" ad_hoc_property="42">\n  <testsuite name="SuccessfulTest" tests="1" failures="0" disabled="0" skipped="0" errors="0" time="*" timestamp="*">\n    <testcase name="Succeeds" file="gtest_xml_output_unittest_.cc" line="53" status="run" result="completed" time="*" timestamp="*" classname="SuccessfulTest"/>\n  </testsuite>\n  <testsuite name="PropertyRecordingTest" tests="1" failures="0" disabled="0" skipped="0" errors="0" time="*" timestamp="*" SetUpTestSuite="yes" TearDownTestSuite="aye">\n    <testcase name="IntValuedProperty" file="gtest_xml_output_unittest_.cc" line="125" status="run" result="completed" time="*" timestamp="*" classname="PropertyRecordingTest">\n      <properties>\n        <property name="key_int" value="1"/>\n      </properties>\n    </testcase>\n  </testsuite>\n  <testsuite name="Single/ValueParamTest" tests="1" failures="0" disabled="0" skipped="0" errors="0" time="*" timestamp="*">\n    <testcase name="HasValueParamAttribute/0" file="gtest_xml_output_unittest_.cc" line="164" value_param="33" status="run" result="completed" time="*" timestamp="*" classname="Single/ValueParamTest" />\n  </testsuite>\n</testsuites>'
EXPECTED_NO_TEST_XML = '<?xml version="1.0" encoding="UTF-8"?>\n<testsuites tests="0" failures="0" disabled="0" errors="0" time="*"\n            timestamp="*" name="AllTests">\n  <testsuite name="NonTestSuiteFailure" tests="1" failures="1" disabled="0" skipped="0" errors="0" time="*" timestamp="*">\n    <testcase name="" status="run" result="completed" time="*" timestamp="*" classname="">\n      <failure message="gtest_no_test_unittest.cc:*&#x0A;Expected equality of these values:&#x0A;  1&#x0A;  2%(stack_entity)s" type=""><![CDATA[gtest_no_test_unittest.cc:*\nExpected equality of these values:\n  1\n  2%(stack)s]]></failure>\n    </testcase>\n  </testsuite>\n</testsuites>' % {'stack': STACK_TRACE_TEMPLATE, 'stack_entity': STACK_TRACE_ENTITY_TEMPLATE}
GTEST_PROGRAM_PATH = gtest_test_utils.GetTestExecutablePath(GTEST_PROGRAM_NAME)
SUPPORTS_TYPED_TESTS = 'TypedTest' in gtest_test_utils.Subprocess([GTEST_PROGRAM_PATH, GTEST_LIST_TESTS_FLAG], capture_stderr=False).output

class GTestXMLOutputUnitTest(gtest_xml_test_utils.GTestXMLTestCase):
    """Unit test for Google Test's XML output functionality."""
    if SUPPORTS_TYPED_TESTS:

        def testNonEmptyXmlOutput(self):
            """Generates non-empty XML and verifies it matches the expected output.

      Runs a test program that generates a non-empty XML output, and
      tests that the XML output is expected.
      """
            self._TestXmlOutput(GTEST_PROGRAM_NAME, EXPECTED_NON_EMPTY_XML, 1)

    def testNoTestXmlOutput(self):
        """Verifies XML output for a Google Test binary without actual tests.

    Runs a test program that generates an XML output for a binary without tests,
    and tests that the XML output is expected.
    """
        self._TestXmlOutput('gtest_no_test_unittest', EXPECTED_NO_TEST_XML, 0)

    def testTimestampValue(self):
        """Checks whether the timestamp attribute in the XML output is valid.

    Runs a test program that generates an empty XML output, and checks if
    the timestamp attribute in the testsuites tag is valid.
    """
        actual = self._GetXmlOutput('gtest_no_test_unittest', [], {}, 0)
        date_time_str = actual.documentElement.getAttributeNode('timestamp').value
        match = re.match('(\\d+)-(\\d\\d)-(\\d\\d)T(\\d\\d):(\\d\\d):(\\d\\d)', date_time_str)
        self.assertTrue(re.match, 'XML datettime string %s has incorrect format' % date_time_str)
        date_time_from_xml = datetime.datetime(year=int(match.group(1)), month=int(match.group(2)), day=int(match.group(3)), hour=int(match.group(4)), minute=int(match.group(5)), second=int(match.group(6)))
        time_delta = abs(datetime.datetime.now() - date_time_from_xml)
        self.assertLess(time_delta, datetime.timedelta(seconds=600))
        actual.unlink()

    def testDefaultOutputFile(self):
        """Tests XML file with default name is created when name is not specified.

    Confirms that Google Test produces an XML output file with the expected
    default name if no name is explicitly specified.
    """
        output_file = os.path.join(gtest_test_utils.GetTempDir(), GTEST_DEFAULT_OUTPUT_FILE)
        gtest_prog_path = gtest_test_utils.GetTestExecutablePath('gtest_no_test_unittest')
        try:
            os.remove(output_file)
        except OSError:
            e = sys.exc_info()[1]
            if e.errno != errno.ENOENT:
                raise
        p = gtest_test_utils.Subprocess([gtest_prog_path, '%s=xml' % GTEST_OUTPUT_FLAG], working_dir=gtest_test_utils.GetTempDir())
        self.assertTrue(p.exited)
        self.assertEqual(0, p.exit_code)
        self.assertTrue(os.path.isfile(output_file))

    def testSuppressedXmlOutput(self):
        """Verifies XML output is suppressed if default listener is shut down.

    Tests that no XML file is generated if the default XML listener is
    shut down before RUN_ALL_TESTS is invoked.
    """
        xml_path = os.path.join(gtest_test_utils.GetTempDir(), GTEST_PROGRAM_NAME + 'out.xml')
        if os.path.isfile(xml_path):
            os.remove(xml_path)
        command = [GTEST_PROGRAM_PATH, '%s=xml:%s' % (GTEST_OUTPUT_FLAG, xml_path), '--shut_down_xml']
        p = gtest_test_utils.Subprocess(command)
        if p.terminated_by_signal:
            self.assertFalse(p.terminated_by_signal, '%s was killed by signal %d' % (GTEST_PROGRAM_NAME, p.signal))
        else:
            self.assertTrue(p.exited)
            self.assertEqual(1, p.exit_code, "'%s' exited with code %s, which doesn't match the expected exit code %s." % (command, p.exit_code, 1))
        self.assertFalse(os.path.isfile(xml_path))

    def testFilteredTestXmlOutput(self):
        """Verifies XML output when a filter is applied.

    Runs a test program that executes only some tests and verifies that
    non-selected tests do not show up in the XML output.
    """
        self._TestXmlOutput(GTEST_PROGRAM_NAME, EXPECTED_FILTERED_TEST_XML, 0, extra_args=['%s=SuccessfulTest.*' % GTEST_FILTER_FLAG])

    def testShardedTestXmlOutput(self):
        """Verifies XML output when run using multiple shards.

    Runs a test program that executes only one shard and verifies that tests
    from other shards do not show up in the XML output.
    """
        self._TestXmlOutput(GTEST_PROGRAM_NAME, EXPECTED_SHARDED_TEST_XML, 0, extra_env={SHARD_INDEX_ENV_VAR: '0', TOTAL_SHARDS_ENV_VAR: '10'})

    def _GetXmlOutput(self, gtest_prog_name, extra_args, extra_env, expected_exit_code):
        """Returns the XML output generated by running the program gtest_prog_name.

    Furthermore, the program's exit code must be expected_exit_code.

    Args:
      gtest_prog_name: Program to run.
      extra_args: Optional arguments to pass to program.
      extra_env: Optional environment variables to set.
      expected_exit_code: Expected exit code from running gtest_prog_name.
    """
        xml_path = os.path.join(gtest_test_utils.GetTempDir(), gtest_prog_name + 'out.xml')
        gtest_prog_path = gtest_test_utils.GetTestExecutablePath(gtest_prog_name)
        command = [gtest_prog_path, '%s=xml:%s' % (GTEST_OUTPUT_FLAG, xml_path)] + extra_args
        environ_copy = os.environ.copy()
        if extra_env:
            environ_copy.update(extra_env)
        p = gtest_test_utils.Subprocess(command, env=environ_copy)
        if p.terminated_by_signal:
            self.assertTrue(False, '%s was killed by signal %d' % (gtest_prog_name, p.signal))
        else:
            self.assertTrue(p.exited)
            self.assertEqual(expected_exit_code, p.exit_code, "'%s' exited with code %s, which doesn't match the expected exit code %s." % (command, p.exit_code, expected_exit_code))
        actual = minidom.parse(xml_path)
        return actual

    def _TestXmlOutput(self, gtest_prog_name, expected_xml, expected_exit_code, extra_args=None, extra_env=None):
        """Asserts that the XML document matches.

    Asserts that the XML document generated by running the program
    gtest_prog_name matches expected_xml, a string containing another
    XML document.  Furthermore, the program's exit code must be
    expected_exit_code.

    Args:
      gtest_prog_name: Program to run.
      expected_xml: Path to XML document to match.
      expected_exit_code: Expected exit code from running gtest_prog_name.
      extra_args: Optional arguments to pass to program.
      extra_env: Optional environment variables to set.
    """
        actual = self._GetXmlOutput(gtest_prog_name, extra_args or [], extra_env or {}, expected_exit_code)
        expected = minidom.parseString(expected_xml)
        self.NormalizeXml(actual.documentElement)
        self.AssertEquivalentNodes(expected.documentElement, actual.documentElement)
        expected.unlink()
        actual.unlink()
if __name__ == '__main__':
    os.environ['GTEST_STACK_TRACE_DEPTH'] = '1'
    gtest_test_utils.Main()