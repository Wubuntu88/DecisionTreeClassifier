import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;


public class DecisionTreeClassifier {
	
	private class Record {
		double[] attrList;
		String label;

		public Record(double[] attrList, String label) {
			this.attrList = attrList;
			this.label = label;
		}
		
		public int numberOfAttributes(){
			return attrList.length;
		}

		@Override
		public String toString() {
			StringBuffer sBuffer = new StringBuffer("");
			for (double dub : this.attrList) {
				sBuffer.append(String.format("%.2f", dub) + ", ");
			}
			sBuffer.replace(sBuffer.length() - 2, sBuffer.length(), " || ");
			sBuffer.append(this.label);
			return sBuffer.toString();
		}
	}//end of Record class
	
	private class TreeNode {
		public static final String INTERNAL = "internal";
		public static final String LEAF = "leaf";
		
		private String nodeType;
		private int attribute; //column index
		private String labelName;
		private TreeNode left;
		private TreeNode right;
		
		public TreeNode(String type, String labelName, int attribute, TreeNode left, TreeNode right){
			nodeType = type;
			this.left = left;
			this.right = right;
			this.attribute = attribute;
			this.labelName = labelName;
		}
		
		@Override
		public String toString() {
			String contents = nodeType.equals(TreeNode.INTERNAL) ? "attribute: " + attribute : "label: " + labelName;
			return "Type: " + nodeType + ", " + contents;
		}
	}
	
	/**************************************************************
	 * MAIN METHOD
	 **************************************************************/
	public static void main(String[] args) {
		/*
		 * **PART 1********************************************
		 * In this part I run the decision tree algorithm on the train1 and test1 data
		 */
		String trainFileName1 = "train1";
		String testFileName1 = "test1";
		String outputFile1 = "output1";
		DecisionTreeClassifier dtc = new DecisionTreeClassifier();
		try {
			dtc.loadTrainingData(trainFileName1);
		} catch (Exception e) {
			e.printStackTrace();
		}
		/* C: Trace of the tree building */
		dtc.buildTree();
		/* A:  */
		//what am I supposed to print...?
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
		
		/* F: compute validation random sampling and one out*/
		double oneOutError = dtc.oneOutValidationError();
		System.out.println("one out error: " + oneOutError);
		double randomSamplingError = dtc.randomSamplingClassificationError();
		System.out.println("random sampling error: " + randomSamplingError);
		
		/*the rest of part one */
		
		/* 
		 * Part 2: runing the decision tree algorithm on train 2 and test 2 ****
		 * #3
		 * */
		dtc = new DecisionTreeClassifier();// start over and make a new decision tree
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
		
		/* C: compute validation random sampling and one out*/
		double oneOutError2 = dtc.oneOutValidationError();
		System.out.println("one out error: " + oneOutError2);
		
		/* D: tree picture */
		System.out.println(dtc.toString());
		
	}
	
	private int numberOfRecords;
	private int numberOfAttributes;
	private int numberOfClasses;

	// list of the type of the variables in records (ordinal, continuous, etc)
	private ArrayList<Record> records = new ArrayList<>();
	private HashMap<String, Integer> binaryNameToIntSymbol = new HashMap<>();
	
	//variables pertaining to the tree
	private TreeNode root;
	private final int RIGHT_RECORDS_BINARY_VALUE = 0;
	private final int LEFT_RECORDS_BINARY_VALUE = 1;
	
	public void buildTree(){
		TreeSet<Integer> indicesOfRecords = new TreeSet<>();
		for(int i = 0; i < records.size(); i++){
			indicesOfRecords.add(i);
		}
		TreeSet<Integer> remainingColIndices = new TreeSet<>();
		for(int i = 0; i < numberOfAttributes; i++){
			remainingColIndices.add(i);
		}
		System.out.println("^^Tree Building^^");
		root = build(indicesOfRecords, remainingColIndices);
	}
	
	/**
	 * Builds a decision tree recursively.  Ultimately returns the root of the tree.
	 * @param indicesOfRecordsLeft
	 * @param remainingColIndices
	 * @return the root TreeNode of the tree
	 */
	public TreeNode build(TreeSet<Integer> indicesOfRecordsLeft, TreeSet<Integer> remainingColIndices){
		System.out.println("-$$-Tree Building Iteration-$$-");
		System.out.println("indicesOfRecordsLeft: " + indicesOfRecordsLeft);
		System.out.println("remainingColumnIndices: " + remainingColIndices);
		if(areRecordsSameClass(indicesOfRecordsLeft)){
			String labelName = records.get(indicesOfRecordsLeft.first()).label;
			System.out.println("--Leaf node created-- Label: " + labelName);
			return new TreeNode(TreeNode.LEAF, labelName, -1,null, null);
		}else if(remainingColIndices.size() == 0 || indicesOfRecordsLeft.size() < 6){
			String majorityLabel = majorityLabel(indicesOfRecordsLeft);
			System.out.println("--Leaf node created-- Label: " + majorityLabel);
			return new TreeNode(TreeNode.LEAF, majorityLabel, -1, null, null);
		}else{//the real tree building
			int bestColumnToSplitRecords = bestColIndexToSplitRecords(indicesOfRecordsLeft, remainingColIndices);
			TreeSet<Integer> leftIndices = 
					indicesOfRecordsWithValueAtColumn(indicesOfRecordsLeft, 
							LEFT_RECORDS_BINARY_VALUE, bestColumnToSplitRecords);
			TreeSet<Integer> rightIndices = 
					indicesOfRecordsWithValueAtColumn(indicesOfRecordsLeft, 
							RIGHT_RECORDS_BINARY_VALUE, bestColumnToSplitRecords);
			if(leftIndices.size() == 0 || rightIndices.size() == 0){
				String label = majorityLabel(indicesOfRecordsLeft);
				System.out.println("--Leaf node created-- Label: " + label);
				return new TreeNode(TreeNode.LEAF, label, -1, null, null);
			}else{//building left and right nodes
				TreeSet<Integer> leftRemainingColIndices = new TreeSet<>(remainingColIndices);
				TreeSet<Integer> rightRemainingColIndices = new TreeSet<>(remainingColIndices);
				
				leftRemainingColIndices.remove(bestColumnToSplitRecords);
				rightRemainingColIndices.remove(bestColumnToSplitRecords);
				
				System.out.println("**Internal node created** Best condition: " + bestColumnToSplitRecords);
				System.out.println("left node indices passed to subtree: " + leftIndices);
				System.out.println("right node indices passed to subtree: " + rightIndices);
				
				TreeNode left = build(leftIndices, leftRemainingColIndices);
				TreeNode right = build(rightIndices, rightRemainingColIndices);
				
				TreeNode node = new TreeNode(TreeNode.INTERNAL, null, bestColumnToSplitRecords, left, right);
				return node;
			}
		}
	}
	/**
	 * Returns the label with the highest frequency from the records specified by the indices parameter
	 * @param indicesOfRecordsLeft
	 * @return majority label
	 */
	private String majorityLabel(TreeSet<Integer> indicesOfRecordsLeft) {
		HashMap<String, Integer> frequencyOfLabels = new HashMap<>();
		for(Integer indexOfRecord: indicesOfRecordsLeft){
			String labelOfRecord = records.get(indexOfRecord).label;
			if(frequencyOfLabels.containsKey(labelOfRecord)){
				frequencyOfLabels.put(labelOfRecord, frequencyOfLabels.get(labelOfRecord) + 1);
			}else{//if the key is not in the hashmap yet
				frequencyOfLabels.put(labelOfRecord, 1);
			}
		}
		String maxLabel = null;
		int maxFrequency = Integer.MIN_VALUE;
		for(String label: frequencyOfLabels.keySet()){
			int frequencyOfCurrentLabel = frequencyOfLabels.get(label);
			if(frequencyOfLabels.get(label) > maxFrequency){
				maxFrequency = frequencyOfCurrentLabel;
				maxLabel = label;
			}
		}
		return maxLabel;
	}
	/**
	 * @param indicesOfRecords
	 * @return True if all of the records have the same class; false otherwise.
	 */
	private boolean areRecordsSameClass(TreeSet<Integer> indicesOfRecords){
		if(indicesOfRecords.size() == 0){
			return false;
		}
		String classOfRecord = records.get(indicesOfRecords.last()).label;
		for(Integer indexOfRecord: indicesOfRecords){
			Record record = records.get(indexOfRecord);
			if(record.label.equals(classOfRecord) == false){
				return false;
			}
		}
		return true;
	}
	/**
	 * Loads the training data into the Ivars: numberOfRecords, numberOfAttributes, etc.
	 * Also fills the records Ivar.
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
			//String typeOfAttrAtIndex = this.attributeList.get(colIndex);
			for(String binaryName: strRange){
					binaryNameToIntSymbol.put(binaryName, binaryVariableCounter);
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
			for(int colIndex = 0; colIndex < comps.length - 1; colIndex++){
				String stringValAtColIndex = comps[colIndex];
				attrs[colIndex] = (double)binaryNameToIntSymbol.get(stringValAtColIndex);
			}
			Record recordToAdd = new Record(attrs, label);
			records.add(recordToAdd);
		}
	}// end of loadTrainingData()
	/**
	 * 
	 * @param fileName
	 * @return An arrayList of the records (label is null)
	 * @throws IOException
	 */
	public ArrayList<Record> loadTestRecordsFromFile(String fileName) throws IOException{
		String whitespace = "[ ]+";
		List<String> lines = Files.readAllLines(Paths.get(fileName),
				Charset.defaultCharset());
		ArrayList<Record> testRecords = new ArrayList<>();
		for(String line: lines){
			String[] comps = line.split(whitespace);
			double[] attrs = new double[comps.length];
			for(int i = 0; i < comps.length; i++){
				attrs[i] = binaryNameToIntSymbol.get(comps[i]);
			}
			Record recordToAdd = new Record(attrs, null);
			testRecords.add(recordToAdd);
		}
		return testRecords;
	}
	/**
	 * Takes An array of classified records and filename as input and write the classified
	 * records' labels to an input file specified by the user.
	 * @param fileName
	 * @param theRecords
	 */
	public void classifyTestRecordsAndWriteToFile(String fileName, ArrayList<Record> theRecords){
		PrintWriter pw = null;
		try {
			pw = new PrintWriter(fileName);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		StringBuffer sBuffer = new StringBuffer("");
		for(Record theRecord: theRecords){
			String label = classify(theRecord);
			sBuffer.append(label + "\n");
		}
		sBuffer.delete(sBuffer.length() - 1, sBuffer.length());
		pw.write(sBuffer.toString());
		pw.close();
	}
	/**
	 * @param indicesOfRecords
	 * @param remainingColIndices
	 * @return Returns the column index of the best attribute on which to split the records.
	 */
	private Integer bestColIndexToSplitRecords(Set<Integer> indicesOfRecords, Set<Integer> remainingColIndices){
		if(remainingColIndices.size() == 0 || indicesOfRecords.size() == 0){
			return null;
		}
		double minValue = Double.MAX_VALUE;
		Integer bestColumnIndex = null;//the column index has the lowest average weighted entropy
		for(Integer colIndex: remainingColIndices){
			double value = averageWeightedEntropy(indicesOfRecords, colIndex);
			if (value < minValue){
				minValue = value;
				bestColumnIndex = colIndex;
			}
		}
		return bestColumnIndex;
	}
	/**
	 * @param indicesOfRecords
	 * @param colIndex
	 * @return returns the average weighted entropy
	 */
	private double averageWeightedEntropy(Set<Integer> indicesOfRecords, int colIndex){
		
		TreeSet<Integer> leftRecords = indicesOfRecordsWithValueAtColumn(indicesOfRecords, 
				LEFT_RECORDS_BINARY_VALUE, colIndex);
		HashMap<String, Integer> freqOfBinValuesLeftRecords = frequenciesOfLabels(leftRecords, colIndex);
		TreeSet<Integer> rightRecords = indicesOfRecordsWithValueAtColumn(indicesOfRecords, 
				RIGHT_RECORDS_BINARY_VALUE, colIndex);
		HashMap<String, Integer> freqOfBinValuesRightRecords = frequenciesOfLabels(rightRecords, colIndex);

		double entropyLeft = gini(freqOfBinValuesLeftRecords, leftRecords.size());
		double entropyRight = gini(freqOfBinValuesRightRecords, rightRecords.size());

		double average = entropyLeft*leftRecords.size() / indicesOfRecords.size() +
				entropyRight * rightRecords.size() / indicesOfRecords.size();
		
		return average;
	}
	/**
	 * Examines all of the records and returns the set of indices where the record
	 * has the value at that column index
	 * @param indicesOfRecords
	 * @param value
	 * @param colIndex
	 * @return A TreeSet of indices of records with a value at a column index
	 */
	private TreeSet<Integer> indicesOfRecordsWithValueAtColumn(Set<Integer> indicesOfRecords,
																	int value, int colIndex){
		TreeSet<Integer> setOfIndicesToReturn = new TreeSet<>();
		for(Integer index: indicesOfRecords){
			Record recordAtRow = records.get(index);
			if((int)recordAtRow.attrList[colIndex] == value){
				setOfIndicesToReturn.add(index);
			}
		}
		return setOfIndicesToReturn;
	}
	/**
	 * Calculates the frequency of labels for a specified set of indices of records.
	 * @param indicesOfRecords
	 * @param colIndex
	 * @return A hashmap where the key is a label and the value is its frequency.
	 */
	private HashMap<String, Integer> frequenciesOfLabels(TreeSet<Integer> indicesOfRecords, int colIndex){
		HashMap<String, Integer> frequencyOfLabels = new HashMap<>();
		for(Integer rowIndex: indicesOfRecords){
			Record record = records.get(rowIndex);
			String labelKey = record.label;
			if(frequencyOfLabels.containsKey(labelKey)){
				frequencyOfLabels.put(labelKey, frequencyOfLabels.get(labelKey) + 1);
			}else{//if the key is not in the hashmap yet
				frequencyOfLabels.put(labelKey, 1);
			}
		}
		return frequencyOfLabels;
	}

	private static double entropy(HashMap<String, Integer> frequencyOfBinaryValues, int size){
		double entropySoFar = 0.0;
		for(String binaryKey: frequencyOfBinaryValues.keySet()){
			double ratio = (double)frequencyOfBinaryValues.get(binaryKey) / size;
			entropySoFar += ratio * Math.log(ratio);
		}
		return -entropySoFar;//negative of entropySoFar
	}
	
	private static double gini(HashMap<String, Integer> frequencyOfBinaryValues, int size){
		double giniFigureSoFar = 0.0;
		for(String binaryKey: frequencyOfBinaryValues.keySet()){
			giniFigureSoFar += Math.pow((double)frequencyOfBinaryValues.get(binaryKey) / size, 2);
		}
		return 1 - giniFigureSoFar;
	}
	
	private static double classError(HashMap<String, Integer> frequencyOfBinaryValues, int size){
		double maxClassError = Double.MIN_VALUE;
		for (String binaryKey : frequencyOfBinaryValues.keySet()) {
			double ratio = (double)frequencyOfBinaryValues.get(binaryKey) / size;
			if(ratio > maxClassError){
				maxClassError = ratio;
			}
		}
		return 1 - maxClassError;
	}
	
	public String classify(Record recordToClassify){
		TreeNode currentNode = root;
		while(currentNode.nodeType.equals(TreeNode.INTERNAL)){
			int colIndexForSplitting = currentNode.attribute;
			if(recordToClassify.attrList[colIndexForSplitting] == RIGHT_RECORDS_BINARY_VALUE){
				currentNode = currentNode.right;
			}else{
				currentNode = currentNode.left;
			}
		}
		return currentNode.labelName;
	}
	
	public double computeTrainingError(){
		int numberOfMisclassifiedRecords = 0;
		for(Record currentRecord: records){
			String labelOfClassifiedRecord = classify(currentRecord);
			if(currentRecord.label.equals(labelOfClassifiedRecord) == false){
				numberOfMisclassifiedRecords++;
			}
		}
		return (double)numberOfMisclassifiedRecords / records.size();
	}
	
	public double randomSamplingClassificationError(){
		//15% of the records will be used as test records; 85% for training.
		Random rng = new Random();
		int numberOfTestRecords = (int)(records.size() * 0.15);
		int numberOfMisclassifiedRecords = 0;
		int ITERATIONS = 300;
		for(int iteration = 0; iteration < ITERATIONS; iteration++){
			ArrayList<Record> testRecords = new ArrayList<>();
			for(int i = 0; i < numberOfTestRecords; i++){
				int randomIndex = rng.nextInt(records.size());
				testRecords.add(records.remove(randomIndex));
			}
			
			buildTree();
			for(Record testRecord: testRecords){
				String label = classify(testRecord);
				if(label.equals(testRecord.label) == false){
					numberOfMisclassifiedRecords++;
				}
			}
			
			records.addAll(testRecords);
		}
		return (double)numberOfMisclassifiedRecords / (numberOfTestRecords * ITERATIONS);
	}
	
	public double oneOutValidationError(){
		int numberOfMisclassifiedRecords = 0;
		for(int i = 0; i < records.size(); i++){
			Record recordToClassify = records.remove(i);
			buildTree();
			System.out.println(treeString(root, 0));
			String label = classify(recordToClassify);
			//System.out.println("classified label: " + label + ", record's label: " + recordToClassify.label);
			if(label.equals(recordToClassify.label) == false){
				numberOfMisclassifiedRecords++;
			}
			records.add(i, recordToClassify);
		}
		return (double)numberOfMisclassifiedRecords / records.size();
	}
	
	public static void zTestClassErrors(){
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
	/**
	 * Prints the tree sideways
	 * @param node
	 * @param hierarchy
	 * @return a string description of how the tree looks sideways.
	 */
	public String treeString(TreeNode node, int hierarchy){
		if(node == null){
			return "";
		}else{
			StringBuffer sBuffer = new StringBuffer("");
			for(int i = 0; i < hierarchy; i++){
				sBuffer.append("\t");
			}
			sBuffer.append(node.toString() + "\n");
			sBuffer.append(treeString(node.left, hierarchy + 1));
			sBuffer.append(treeString(node.right, hierarchy + 1));
			return sBuffer.toString();
		}
	}

	@Override
	public String toString() {
		StringBuffer sBuffer = new StringBuffer("");
		sBuffer.append("# Of Attributes: " + this.numberOfAttributes + "\n");
		sBuffer.append("# Of Classes: " + this.numberOfClasses + "\n");
		sBuffer.append("# Of Records: " + this.numberOfRecords + "\n");		
		String treeDescription = treeString(root, 0);
		sBuffer.append(treeDescription);
		return sBuffer.toString();
	}

}
