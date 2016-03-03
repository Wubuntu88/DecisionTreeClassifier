import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;

public class DecisionTreeClassifier {
	private class TreeNode {
		public static final String INTERNAL = "internal";
		public static final String LEAF = "leaf";

		private String nodeType;
		private int attribute; // column index
		private String labelName;
		private TreeNode left;
		private TreeNode right;

		public TreeNode(String type, String labelName, int attribute,
				TreeNode left, TreeNode right) {
			this.nodeType = type;
			this.left = left;
			this.right = right;
			this.attribute = attribute;
			this.labelName = labelName;
		}

		@Override
		public String toString() {
			String contents = this.nodeType.equals(TreeNode.INTERNAL)
					? "attribute: " + this.attribute
					: "label: " + this.labelName;
			return "Type: " + this.nodeType + ", " + contents;
		}
	}

	public static String SHANNON = "SHANNON";

	public static String CLASS = "CLASS";
	public static String GINI = "GINI";

	private static double classError(
			HashMap<String, Integer> frequencyOfBinaryValues, int size) {
		double maxClassError = Double.MIN_VALUE;
		for (String binaryKey : frequencyOfBinaryValues.keySet()) {
			double ratio = (double) frequencyOfBinaryValues.get(binaryKey)
					/ size;
			if (ratio > maxClassError) {
				maxClassError = ratio;
			}
		}
		return 1 - maxClassError;
	}

	private static double entropy(
			HashMap<String, Integer> frequencyOfBinaryValues, int size) {
		double entropySoFar = 0.0;
		for (String binaryKey : frequencyOfBinaryValues.keySet()) {
			double ratio = (double) frequencyOfBinaryValues.get(binaryKey)
					/ size;
			entropySoFar += ratio * Math.log(ratio);
		}
		return -entropySoFar;// negative of entropySoFar
	}

	private static double gini(HashMap<String, Integer> frequencyOfBinaryValues,
			int size) {
		double giniFigureSoFar = 0.0;
		for (String binaryKey : frequencyOfBinaryValues.keySet()) {
			giniFigureSoFar += Math.pow(
					(double) frequencyOfBinaryValues.get(binaryKey) / size, 2);
		}
		return 1 - giniFigureSoFar;
	}

	public static void zTestClassErrors() {
		HashMap<String, Integer> map = new HashMap<>();
		map.put("0", 16);
		map.put("1", 6);
		int size = 20;

		double entropy = DecisionTreeClassifier.entropy(map, size);
		double gini = DecisionTreeClassifier.gini(map, size);
		double classError = DecisionTreeClassifier.classError(map, size);
		System.out.println("entropy: " + entropy);
		System.out.println("gini: " + gini);
		System.out.println("class error: " + classError);
	}

	private String entropyType;

	private int numberOfRecords;
	private int numberOfAttributes;

	private int numberOfClasses;
	// list of the type of the variables in records (ordinal, continuous, etc)
	private ArrayList<Record> records = new ArrayList<>();
	private HashMap<String, Integer> binaryNameToIntSymbol = new HashMap<>();

	// variables pertaining to the tree
	private TreeNode root;

	private final int RIGHT_RECORDS_BINARY_VALUE = 0;
	private final int LEFT_RECORDS_BINARY_VALUE = 1;

	public DecisionTreeClassifier(String entropyType) {
		this.entropyType = entropyType;
	}

	/**
	 * @param indicesOfRecords
	 * @return True if all of the records have the same class; false otherwise.
	 */
	private boolean areRecordsSameClass(TreeSet<Integer> indicesOfRecords) {
		if (indicesOfRecords.size() == 0) {
			return false;
		}
		String classOfRecord = this.records.get(indicesOfRecords.last()).label;
		for (Integer indexOfRecord : indicesOfRecords) {
			Record record = this.records.get(indexOfRecord);
			if (record.label.equals(classOfRecord) == false) {
				return false;
			}
		}
		return true;
	}

	/**
	 * @param indicesOfRecords
	 * @param colIndex
	 * @return returns the average weighted entropy
	 */
	private double averageWeightedEntropy(Set<Integer> indicesOfRecords,
			int colIndex) {

		TreeSet<Integer> leftRecords = this.indicesOfRecordsWithValueAtColumn(
				indicesOfRecords, this.LEFT_RECORDS_BINARY_VALUE, colIndex);
		HashMap<String, Integer> freqOfBinValuesLeftRecords = this
				.frequenciesOfLabels(leftRecords, colIndex);
		TreeSet<Integer> rightRecords = this.indicesOfRecordsWithValueAtColumn(
				indicesOfRecords, this.RIGHT_RECORDS_BINARY_VALUE, colIndex);
		HashMap<String, Integer> freqOfBinValuesRightRecords = this
				.frequenciesOfLabels(rightRecords, colIndex);
		double entropyLeft;
		double entropyRight;
		if (this.entropyType.equals(DecisionTreeClassifier.CLASS)) {
			entropyLeft = classError(freqOfBinValuesLeftRecords,
					leftRecords.size());
			entropyRight = classError(freqOfBinValuesRightRecords,
					rightRecords.size());
		} else if (this.entropyType.equals(DecisionTreeClassifier.SHANNON)) {
			entropyLeft = entropy(freqOfBinValuesLeftRecords,
					leftRecords.size());
			entropyRight = entropy(freqOfBinValuesRightRecords,
					rightRecords.size());
		} else {
			entropyLeft = gini(freqOfBinValuesLeftRecords, leftRecords.size());
			entropyRight = gini(freqOfBinValuesRightRecords,
					rightRecords.size());
		}

		double average = entropyLeft * leftRecords.size()
				/ indicesOfRecords.size()
				+ entropyRight * rightRecords.size() / indicesOfRecords.size();

		return average;
	}

	/**
	 * @param indicesOfRecords
	 * @param remainingColIndices
	 * @return Returns the column index of the best attribute on which to split
	 *         the records.
	 */
	private Integer bestColIndexToSplitRecords(Set<Integer> indicesOfRecords,
			Set<Integer> remainingColIndices) {
		if (remainingColIndices.size() == 0 || indicesOfRecords.size() == 0) {
			return null;
		}
		double minValue = Double.MAX_VALUE;
		Integer bestColumnIndex = null;// the column index has the lowest
										// average weighted entropy
		for (Integer colIndex : remainingColIndices) {
			double value = this.averageWeightedEntropy(indicesOfRecords,
					colIndex);
			if (value < minValue) {
				minValue = value;
				bestColumnIndex = colIndex;
			}
		}
		return bestColumnIndex;
	}

	/**
	 * Builds a decision tree recursively. Ultimately returns the root of the
	 * tree.
	 *
	 * @param indicesOfRecordsLeft
	 * @param remainingColIndices
	 * @return the root TreeNode of the tree
	 */
	public TreeNode build(TreeSet<Integer> indicesOfRecordsLeft,
			TreeSet<Integer> remainingColIndices) {
		System.out.println("-$$-Tree Building Iteration-$$-");
		System.out.println("indicesOfRecordsLeft: " + indicesOfRecordsLeft);
		System.out.println("remainingColumnIndices: " + remainingColIndices);
		if (this.areRecordsSameClass(indicesOfRecordsLeft)) {
			String labelName = this.records
					.get(indicesOfRecordsLeft.first()).label;
			System.out.println("--Leaf node created-- Label: " + labelName);
			return new TreeNode(TreeNode.LEAF, labelName, -1, null, null);
		} else if (remainingColIndices.size() == 0
				|| indicesOfRecordsLeft.size() < 6) {
			String majorityLabel = this.majorityLabel(indicesOfRecordsLeft);
			System.out.println("--Leaf node created-- Label: " + majorityLabel);
			return new TreeNode(TreeNode.LEAF, majorityLabel, -1, null, null);
		} else {// the real tree building
			int bestColumnToSplitRecords = this.bestColIndexToSplitRecords(
					indicesOfRecordsLeft, remainingColIndices);
			TreeSet<Integer> leftIndices = this
					.indicesOfRecordsWithValueAtColumn(indicesOfRecordsLeft,
							this.LEFT_RECORDS_BINARY_VALUE,
							bestColumnToSplitRecords);
			TreeSet<Integer> rightIndices = this
					.indicesOfRecordsWithValueAtColumn(indicesOfRecordsLeft,
							this.RIGHT_RECORDS_BINARY_VALUE,
							bestColumnToSplitRecords);
			if (leftIndices.size() == 0 || rightIndices.size() == 0) {
				String label = this.majorityLabel(indicesOfRecordsLeft);
				System.out.println("--Leaf node created-- Label: " + label);
				return new TreeNode(TreeNode.LEAF, label, -1, null, null);
			} else {// building left and right nodes
				TreeSet<Integer> leftRemainingColIndices = new TreeSet<>(
						remainingColIndices);
				TreeSet<Integer> rightRemainingColIndices = new TreeSet<>(
						remainingColIndices);

				leftRemainingColIndices.remove(bestColumnToSplitRecords);
				rightRemainingColIndices.remove(bestColumnToSplitRecords);

				System.out.println("**Internal node created** Best condition: "
						+ bestColumnToSplitRecords);
				System.out.println(
						"left node indices passed to subtree: " + leftIndices);
				System.out.println("right node indices passed to subtree: "
						+ rightIndices);

				TreeNode left = this.build(leftIndices,
						leftRemainingColIndices);
				TreeNode right = this.build(rightIndices,
						rightRemainingColIndices);

				TreeNode node = new TreeNode(TreeNode.INTERNAL, null,
						bestColumnToSplitRecords, left, right);
				return node;
			}
		}
	}

	public void buildTree() {
		TreeSet<Integer> indicesOfRecords = new TreeSet<>();
		for (int i = 0; i < this.records.size(); i++) {
			indicesOfRecords.add(i);
		}
		TreeSet<Integer> remainingColIndices = new TreeSet<>();
		for (int i = 0; i < this.numberOfAttributes; i++) {
			remainingColIndices.add(i);
		}
		System.out.println("^^Tree Building^^");
		this.root = this.build(indicesOfRecords, remainingColIndices);
	}

	public String classify(Record recordToClassify) {
		TreeNode currentNode = this.root;
		while (currentNode.nodeType.equals(TreeNode.INTERNAL)) {
			int colIndexForSplitting = currentNode.attribute;
			if (recordToClassify.attrList[colIndexForSplitting] == this.RIGHT_RECORDS_BINARY_VALUE) {
				currentNode = currentNode.right;
			} else {
				currentNode = currentNode.left;
			}
		}
		return currentNode.labelName;
	}

	/**
	 * Takes An array of classified records and filename as input and write the
	 * classified records' labels to an input file specified by the user.
	 *
	 * @param fileName
	 * @param theRecords
	 */
	public void classifyTestRecordsAndWriteToFile(String fileName,
			ArrayList<Record> theRecords) {
		PrintWriter pw = null;
		try {
			pw = new PrintWriter(fileName);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		StringBuffer sBuffer = new StringBuffer("");
		for (Record theRecord : theRecords) {
			String label = this.classify(theRecord);
			sBuffer.append(label + "\n");
		}
		sBuffer.delete(sBuffer.length() - 1, sBuffer.length());
		pw.write(sBuffer.toString());
		pw.close();
	}

	public double computeTrainingError() {
		int numberOfMisclassifiedRecords = 0;
		for (Record currentRecord : this.records) {
			String labelOfClassifiedRecord = this.classify(currentRecord);
			if (currentRecord.label.equals(labelOfClassifiedRecord) == false) {
				numberOfMisclassifiedRecords++;
			}
		}
		return (double) numberOfMisclassifiedRecords / this.records.size();
	}

	/**
	 * Calculates the frequency of labels for a specified set of indices of
	 * records.
	 *
	 * @param indicesOfRecords
	 * @param colIndex
	 * @return A hashmap where the key is a label and the value is its
	 *         frequency.
	 */
	private HashMap<String, Integer> frequenciesOfLabels(
			TreeSet<Integer> indicesOfRecords, int colIndex) {
		HashMap<String, Integer> frequencyOfLabels = new HashMap<>();
		for (Integer rowIndex : indicesOfRecords) {
			Record record = this.records.get(rowIndex);
			String labelKey = record.label;
			if (frequencyOfLabels.containsKey(labelKey)) {
				frequencyOfLabels.put(labelKey,
						frequencyOfLabels.get(labelKey) + 1);
			} else {// if the key is not in the hashmap yet
				frequencyOfLabels.put(labelKey, 1);
			}
		}
		return frequencyOfLabels;
	}

	/**
	 * Examines all of the records and returns the set of indices where the
	 * record has the value at that column index
	 *
	 * @param indicesOfRecords
	 * @param value
	 * @param colIndex
	 * @return A TreeSet of indices of records with a value at a column index
	 */
	private TreeSet<Integer> indicesOfRecordsWithValueAtColumn(
			Set<Integer> indicesOfRecords, int value, int colIndex) {
		TreeSet<Integer> setOfIndicesToReturn = new TreeSet<>();
		for (Integer index : indicesOfRecords) {
			Record recordAtRow = this.records.get(index);
			if ((int) recordAtRow.attrList[colIndex] == value) {
				setOfIndicesToReturn.add(index);
			}
		}
		return setOfIndicesToReturn;
	}

	/**
	 *
	 * @param fileName
	 * @return An arrayList of the records (label is null)
	 * @throws IOException
	 */
	public ArrayList<Record> loadTestRecordsFromFile(String fileName)
			throws IOException {
		String whitespace = "[ ]+";
		List<String> lines = Files.readAllLines(Paths.get(fileName),
				Charset.defaultCharset());
		ArrayList<Record> testRecords = new ArrayList<>();
		for (String line : lines) {
			String[] comps = line.split(whitespace);
			double[] attrs = new double[comps.length];
			for (int i = 0; i < comps.length; i++) {
				attrs[i] = this.binaryNameToIntSymbol.get(comps[i]);
			}
			Record recordToAdd = new Record(attrs, null);
			testRecords.add(recordToAdd);
		}
		return testRecords;
	}

	/**
	 * Loads the training data into the Ivars: numberOfRecords,
	 * numberOfAttributes, etc. Also fills the records Ivar.
	 *
	 * @param fileName
	 * @throws IOException
	 */
	public void loadTrainingData(String fileName) throws IOException {
		String whitespace = "[ ]+";
		List<String> lines = Files.readAllLines(Paths.get(fileName),
				Charset.defaultCharset());
		// first line
		String[] componentsOfFirstLine = lines.get(0).split(whitespace);

		this.numberOfRecords = Integer.parseInt(componentsOfFirstLine[0]);
		this.numberOfAttributes = Integer.parseInt(componentsOfFirstLine[1]);
		this.numberOfClasses = Integer.parseInt(componentsOfFirstLine[2]);

		int binaryVariableCounter = 0;
		// third line reading the ranges of the attribute types
		// uses rangeAtColumn hash map
		String[] listOfRanges = lines.get(2).split(whitespace);
		for (int colIndex = 0; colIndex < listOfRanges.length - 1; colIndex++) {
			String[] strRange = listOfRanges[colIndex].split(",");
			// String typeOfAttrAtIndex = this.attributeList.get(colIndex);
			for (String binaryName : strRange) {
				this.binaryNameToIntSymbol.put(binaryName,
						binaryVariableCounter);
				binaryVariableCounter++;
			}
			binaryVariableCounter = 0;
		}
		// now I have to get all of the records
		for (int i = 3; i < lines.size(); i++) {
			String line = lines.get(i);
			String[] comps = line.split(whitespace);
			double[] attrs = new double[comps.length - 1];
			String label = comps[comps.length - 1];
			for (int colIndex = 0; colIndex < comps.length - 1; colIndex++) {
				String stringValAtColIndex = comps[colIndex];
				attrs[colIndex] = this.binaryNameToIntSymbol
						.get(stringValAtColIndex);
			}
			Record recordToAdd = new Record(attrs, label);
			this.records.add(recordToAdd);
		}
	}// end of loadTrainingData()

	/**
	 * Returns the label with the highest frequency from the records specified
	 * by the indices parameter
	 *
	 * @param indicesOfRecordsLeft
	 * @return majority label
	 */
	private String majorityLabel(TreeSet<Integer> indicesOfRecordsLeft) {
		HashMap<String, Integer> frequencyOfLabels = new HashMap<>();
		for (Integer indexOfRecord : indicesOfRecordsLeft) {
			String labelOfRecord = this.records.get(indexOfRecord).label;
			if (frequencyOfLabels.containsKey(labelOfRecord)) {
				frequencyOfLabels.put(labelOfRecord,
						frequencyOfLabels.get(labelOfRecord) + 1);
			} else {// if the key is not in the hashmap yet
				frequencyOfLabels.put(labelOfRecord, 1);
			}
		}
		String maxLabel = null;
		int maxFrequency = Integer.MIN_VALUE;
		for (String label : frequencyOfLabels.keySet()) {
			int frequencyOfCurrentLabel = frequencyOfLabels.get(label);
			if (frequencyOfLabels.get(label) > maxFrequency) {
				maxFrequency = frequencyOfCurrentLabel;
				maxLabel = label;
			}
		}
		return maxLabel;
	}

	public double oneOutValidationError() {
		int numberOfMisclassifiedRecords = 0;
		for (int i = 0; i < this.records.size(); i++) {
			Record recordToClassify = this.records.remove(i);
			this.buildTree();
			System.out.println(this.treeString(this.root, 0));
			String label = this.classify(recordToClassify);
			if (label.equals(recordToClassify.label) == false) {
				numberOfMisclassifiedRecords++;
			}
			this.records.add(i, recordToClassify);
		}
		return (double) numberOfMisclassifiedRecords / this.records.size();
	}

	public double randomSamplingClassificationError() {
		// 15% of the records will be used as test records; 85% for training.
		Random rng = new Random();
		int numberOfTestRecords = (int) (this.records.size() * 0.15);
		int numberOfMisclassifiedRecords = 0;
		int ITERATIONS = 100;
		for (int iteration = 0; iteration < ITERATIONS; iteration++) {
			ArrayList<Record> testRecords = new ArrayList<>();
			for (int i = 0; i < numberOfTestRecords; i++) {
				int randomIndex = rng.nextInt(this.records.size());
				testRecords.add(this.records.remove(randomIndex));
			}

			this.buildTree();
			for (Record testRecord : testRecords) {
				String label = this.classify(testRecord);
				if (label.equals(testRecord.label) == false) {
					numberOfMisclassifiedRecords++;
				}
			}

			this.records.addAll(testRecords);
		}
		return (double) numberOfMisclassifiedRecords
				/ (numberOfTestRecords * ITERATIONS);
	}

	@Override
	public String toString() {
		StringBuffer sBuffer = new StringBuffer("");
		sBuffer.append("# Of Attributes: " + this.numberOfAttributes + "\n");
		sBuffer.append("# Of Classes: " + this.numberOfClasses + "\n");
		sBuffer.append("# Of Records: " + this.numberOfRecords + "\n");
		String treeDescription = this.treeString(this.root, 0);
		sBuffer.append(treeDescription);
		return sBuffer.toString();
	}

	/**
	 * Prints the tree sideways
	 *
	 * @param node
	 * @param hierarchy
	 * @return a string description of how the tree looks sideways.
	 */
	public String treeString(TreeNode node, int hierarchy) {
		if (node == null) {
			return "";
		} else {
			StringBuffer sBuffer = new StringBuffer("");
			for (int i = 0; i < hierarchy; i++) {
				sBuffer.append("\t");
			}
			sBuffer.append(node.toString() + "\n");
			sBuffer.append(this.treeString(node.left, hierarchy + 1));
			sBuffer.append(this.treeString(node.right, hierarchy + 1));
			return sBuffer.toString();
		}
	}

}
