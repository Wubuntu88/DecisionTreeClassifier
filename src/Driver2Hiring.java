import java.io.IOException;
import java.util.ArrayList;

public class Driver2Hiring {
	public static void main(String[] args) {
		/*
		 * Part 2: runing the decision tree algorithm on train 2 and test 2 ****
		 * #3
		 */
		DecisionTreeClassifier dtc = new DecisionTreeClassifier();// start over and make a new decision
											// tree
		String trainFileName2 = "train2";
		String testFileName2 = "test2";
		String outputFile2 = "output2";
		dtc = new DecisionTreeClassifier();
		try {
			dtc.loadTrainingData(trainFileName2);
		} catch (Exception e) {
			e.printStackTrace();
		}
		dtc.buildTree();
		/* A: classification of records in test file */
		ArrayList<Record> testRecords2 = null;
		try {
			testRecords2 = dtc.loadTestRecordsFromFile(testFileName2);
			dtc.classifyTestRecordsAndWriteToFile(outputFile2, testRecords2);
		} catch (IOException e) {
			e.printStackTrace();
		}

		/* B: compute and print training error */
		double trainingError2 = dtc.computeTrainingError();
		System.out.println("trainingError: " + trainingError2);

		/* C: compute validation random sampling and one out */
		double oneOutError2 = dtc.oneOutValidationError();
		System.out.println("one out error: " + oneOutError2);

		/* D: tree picture */
		System.out.println(dtc.toString());
	}
}
