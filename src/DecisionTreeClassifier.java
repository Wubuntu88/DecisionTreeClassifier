import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
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
		
		DecisionTreeClassifier dtc = new DecisionTreeClassifier();
		try {
			dtc.loadTrainingData("train2_use");
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		dtc.buildTree();
		System.out.println(dtc);
		double trainingError = dtc.computeTrainingError();
		System.out.println("trainingError: " + trainingError);
		//System.out.println(dtc.root);
		//System.out.println(dtc.root.left);
		//System.out.println(dtc.root.right);
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
		root = build(indicesOfRecords, remainingColIndices);
	}
	
	public TreeNode build(TreeSet<Integer> indicesOfRecordsLeft, TreeSet<Integer> remainingColIndices){
		//System.out.println("building iteration");
		if(areRecordsSameClass(indicesOfRecordsLeft)){
			String labelName = records.get(0).label;
			return new TreeNode(TreeNode.LEAF, labelName, -1,null, null);
		}else if(remainingColIndices.size() == 0){
			String majorityLabel = majorityLabel(indicesOfRecordsLeft);
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
				return new TreeNode(TreeNode.LEAF, label, -1, null, null);
			}else{//building left and right nodes
				TreeSet<Integer> leftRemainingColIndices = new TreeSet<>(remainingColIndices);
				TreeSet<Integer> rightRemainingColIndices = new TreeSet<>(remainingColIndices);
				
				leftRemainingColIndices.remove(bestColumnToSplitRecords);
				rightRemainingColIndices.remove(bestColumnToSplitRecords);
				
				TreeNode left = build(leftIndices, leftRemainingColIndices);
				TreeNode right = build(rightIndices, rightRemainingColIndices);
				
				TreeNode node = new TreeNode(TreeNode.INTERNAL, null, bestColumnToSplitRecords, left, right);
				return node;
			}
		}
	}
	
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
	
	
	public void loadTrainingData(String fileName) throws IOException {
		String whitespace = "[ ]+";
		List<String> lines = Files.readAllLines(Paths.get(fileName),
				Charset.defaultCharset());
		// first line
		String[] componentsOfFirstLine = lines.get(0).split(whitespace);
		
		this.numberOfRecords = Integer.parseInt(componentsOfFirstLine[0]);
		this.numberOfAttributes = Integer.parseInt(componentsOfFirstLine[1]);
		this.numberOfClasses = Integer.parseInt(componentsOfFirstLine[2]);

		// second line reading the attribute types
		String[] componentsOfThirdLine = lines.get(1).split(whitespace);
		//for binary variables
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
	/*
	public ArrayList<Record> loadTestRecordsFromFile(String fileName) throws IOException{
		String whitespace = "[ ]+";
		List<String> lines = Files.readAllLines(Paths.get(fileName),
				Charset.defaultCharset());
		ArrayList<Record> testRecords = 
		return null;
	}
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
	
	private double averageWeightedEntropy(Set<Integer> indicesOfRecords, int colIndex){
		
		TreeSet<Integer> leftRecords = indicesOfRecordsWithValueAtColumn(indicesOfRecords, 0, colIndex);
		HashMap<Integer, Integer> freqOfBinValuesLeftRecords = frequenciesOfBinaryValues(leftRecords, colIndex);
		TreeSet<Integer> rightRecords = indicesOfRecordsWithValueAtColumn(indicesOfRecords, 1, colIndex);
		HashMap<Integer, Integer> freqOfBinValuesRightRecords = frequenciesOfBinaryValues(rightRecords, colIndex);
		
		double entropyLeft = entropy(freqOfBinValuesLeftRecords, leftRecords.size());
		double entropyRight = entropy(freqOfBinValuesRightRecords, rightRecords.size());
		
		double average = entropyLeft*leftRecords.size() / indicesOfRecords.size() +
				entropyRight * rightRecords.size() / indicesOfRecords.size();
		
		return average;
	}
	
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
	
	private HashMap<Integer, Integer> frequenciesOfBinaryValues(TreeSet<Integer> indicesOfRecords, int colIndex){
		HashMap<Integer, Integer> frequencyOfBinaryValues = new HashMap<>();
		for(Integer rowIndex: indicesOfRecords){
			Record record = records.get(rowIndex);
			int key = (int)record.attrList[colIndex];
			if(frequencyOfBinaryValues.containsKey(key)){
				frequencyOfBinaryValues.put(key, frequencyOfBinaryValues.get(key) + 1);
			}else{//if the key is not in the hashmap yet
				frequencyOfBinaryValues.put(key, 1);
			}
		}
		return frequencyOfBinaryValues;
	}
	
	private static double entropy(HashMap<Integer, Integer> frequencyOfBinaryValues, int size){
		double entropySoFar = 0.0;
		for(Integer binaryKey: frequencyOfBinaryValues.keySet()){
			double ratio = (double)frequencyOfBinaryValues.get(binaryKey) / size;
			entropySoFar += ratio * Math.log(ratio);
		}
		return -entropySoFar;//negative of entropySoFar
	}
	
	private static double gini(HashMap<Integer, Integer> frequencyOfBinaryValues, int size){
		double giniFigureSoFar = 0.0;
		for(Integer binaryKey: frequencyOfBinaryValues.keySet()){
			giniFigureSoFar += Math.pow((double)frequencyOfBinaryValues.get(binaryKey) / size, 2);
		}
		return 1 - giniFigureSoFar;
	}
	
	private static double classError(HashMap<Integer, Integer> frequencyOfBinaryValues, int size){
		double maxClassError = Double.MIN_VALUE;
		for (Integer binaryKey : frequencyOfBinaryValues.keySet()) {
			double ratio = (double)frequencyOfBinaryValues.get(binaryKey) / size;
			if(ratio > maxClassError){
				maxClassError = ratio;
			}
		}
		return 1 - maxClassError;
	}
	
	public double computeTrainingError(){
		int numberOfMisclassifiedRecords = 0;
		for(Record currentRecord: records){
			String labelOfClassifiedRecord = classify(currentRecord);
			System.out.println(currentRecord.label);
			System.out.println(labelOfClassifiedRecord);
			System.out.println();
			if(currentRecord.label.equals(labelOfClassifiedRecord) == false){
				numberOfMisclassifiedRecords++;
			}
		}
		return (double)numberOfMisclassifiedRecords / records.size();
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
	
	public static void zTestClassErrors(){
		HashMap<Integer, Integer> map = new HashMap<>();
		map.put(0, 16);
		map.put(1, 6);
		int size = 20;
		
		double entropy = DecisionTreeClassifier.entropy(map, size);
		double gini = DecisionTreeClassifier.gini(map, size);
		double classError = DecisionTreeClassifier.classError(map, size);
		System.out.println("entropy: " + entropy);
		System.out.println("gini: " + gini);
		System.out.println("class error: " + classError);
	}
	
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
