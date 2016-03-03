import java.io.IOException;
import java.util.ArrayList;

public class Driver1 {
	public static void main(String[] args) {
		/*
		 * **PART 1******************************************** In this part I
		 * run the decision tree algorithm on the train1 and test1 data
		 */
		String trainFileName1 = "train1";
		String testFileName1 = "test1";
		String outputFile1 = "output1";
		DecisionTreeClassifier dtc = new DecisionTreeClassifier();

		try {
			dtc.loadTrainingData(trainFileName1);
		} catch (Exception exp) {
			exp.printStackTrace();
		}
		/* C: Trace of the tree building */
		dtc.buildTree();
		/* A: */
		// what am I supposed to print...?
		/* B: prints the tree structure (albeit sideways) */
		System.out.println(dtc);
		/* D: classify test records and write them to a file */
		ArrayList<Record> testRecords1 = null;
		try {
			testRecords1 = dtc.loadTestRecordsFromFile(testFileName1);
			dtc.classifyTestRecordsAndWriteToFile(outputFile1, testRecords1);
		} catch (IOException e) {
			e.printStackTrace();
		}

		/* E: compute and print training error */
		double trainingError = dtc.computeTrainingError();
		System.out.println("trainingError: " + trainingError);

		/* F: compute validation random sampling and one out */
		double oneOutError = dtc.oneOutValidationError();
		System.out.println("one out error: " + oneOutError);
		double randomSamplingError = dtc.randomSamplingClassificationError();
		System.out.println("random sampling error: " + randomSamplingError);

		/* the rest of part one */
	}
}
