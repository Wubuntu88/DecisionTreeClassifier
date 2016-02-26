import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
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
	}
	
	public static final String BINARY = "binary";
	public static final String CATEGORICAL = "categorical";
	public static final String ORDINAL = "ordinal";
	public static final String CONTINUOUS = "continuous";
	public static final String LABEL = "label";
	private static final TreeSet<String> attributeTypes = new TreeSet<String>(
			Arrays.asList(BINARY, CATEGORICAL, ORDINAL, CONTINUOUS, LABEL));

	public static void main(String[] args) {
		
		DecisionTreeClassifier dtc = new DecisionTreeClassifier();
		try {
			dtc.loadTrainingData("train2_use");
		} catch (Exception e) {
			e.printStackTrace();
		}
		System.out.println(dtc.toString());
		
	}
	
	private int numberOfRecords;
	private int numberOfAttributes;
	private int numberOfClasses;

	// list of the type of the variables in records (ordinal, continuous, etc)
	private ArrayList<String> attributeList = new ArrayList<>();

	private ArrayList<Record> records = new ArrayList<>();

	// for continuous variables (key is column, value is range (array of len 2)
	private HashMap<Integer, double[]> rangeAtColumn = new HashMap<>();
	// for ordinal variables
	private HashMap<Integer, HashMap<String, Double>> valsForOrdinalVarAtColumn = new HashMap<>();
	//for binary variables
	private HashMap<String, Integer> binaryNameToIntSymbol = new HashMap<>();
	//for categorical variables
	private HashMap<String, Integer> categoricalNameToIntSymbol = new HashMap<>();
	
	public void loadTrainingData(String fileName) throws Exception {
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
		for (String attrType : componentsOfThirdLine) {
			if (DecisionTreeClassifier.attributeTypes
					.contains(attrType) == true) {
				this.attributeList.add(attrType);
			} else {
				throw new Exception(
						"attribute in file not one of the correct attributes");
			}
		}
		//for binary variables
		int binaryVariableCounter = 0;
		//for categorical variables
		int categoricalVariableCounter = 0;
		
		// third line reading the ranges of the attribute types
		// uses rangeAtColumn hash map
		String[] listOfRanges = lines.get(2).split(whitespace);
		for (int colIndex = 0; colIndex < listOfRanges.length - 1; colIndex++) {
			// range symbols are low to high
			String[] strRange = listOfRanges[colIndex].split(",");
			double[] range = new double[strRange.length];
			String typeOfAttrAtIndex = this.attributeList.get(colIndex);
			switch (typeOfAttrAtIndex) {
			case ORDINAL:
				// range symbols are low to high
				int index = 0;
				for (String symbol : strRange) {
					range[index] = (double)index / (strRange.length - 1);
					if (this.valsForOrdinalVarAtColumn.containsKey(colIndex)) {
						HashMap<String, Double> map = this.valsForOrdinalVarAtColumn
								.get(colIndex);
						map.put(symbol, range[index]);
					} else {// create the hash map
						HashMap<String, Double> map = new HashMap<>();
						map.put(symbol, range[index]);
						this.valsForOrdinalVarAtColumn.put(colIndex, map);
					}
					index++;
				}
				this.rangeAtColumn.put(colIndex, range);
				break;
			case CONTINUOUS:
				range[0] = Double.parseDouble(strRange[0]);
				range[1] = Double.parseDouble(strRange[1]);
				this.rangeAtColumn.put(colIndex, range);
				break;
			case BINARY:
				for(String binaryName: strRange){
					binaryNameToIntSymbol.put(binaryName, binaryVariableCounter);
					binaryVariableCounter++;
				}
				break;// do later because first file doesn't have binary
			case CATEGORICAL:
				for(String categoricalName: strRange){
					categoricalNameToIntSymbol.put(categoricalName, categoricalVariableCounter);
					categoricalVariableCounter++;
				}
				break;
			}
		}
		// now I have to get all of the records
		for (int i = 3; i < lines.size(); i++) {
			String line = lines.get(i);
			String[] comps = line.split(whitespace);
			double[] attrs = new double[comps.length - 1];
			String label = comps[comps.length - 1];
			for(int colIndex = 0; colIndex < comps.length - 1; colIndex++){
				String stringValAtColIndex = comps[colIndex];
				String typeOfAttr = attributeList.get(colIndex);
				switch (typeOfAttr) {
				case ORDINAL:
					HashMap<String, Double> levelOfOrdinalToDoubleAmount = valsForOrdinalVarAtColumn.get(colIndex);
					double dub = levelOfOrdinalToDoubleAmount.get(stringValAtColIndex);
					attrs[colIndex] = dub;
					break;
				case CONTINUOUS://will have to normalize after
					double amountAtColIndex = Double.parseDouble(stringValAtColIndex);
					attrs[colIndex] = amountAtColIndex;
					break;
				case BINARY:
					attrs[colIndex] = (double)binaryNameToIntSymbol.get(stringValAtColIndex);
					break;
				case CATEGORICAL:
					attrs[colIndex] = (double)categoricalNameToIntSymbol.get(stringValAtColIndex);
					break;
				}
			}
			Record recordToAdd = new Record(attrs, label);
			records.add(recordToAdd);
		}
		//normalizing continuous variables in records so that they range from 0 to 1
		for(Record record: records){
			for(int index = 0; index < record.attrList.length;index++){
				if(attributeList.get(index).equals(CONTINUOUS)){
					double[] rangeAtCol = rangeAtColumn.get(index);
					double max = rangeAtCol[1];
					double min = rangeAtCol[0];
					record.attrList[index] = (record.attrList[index] - min) / (max - min);
				}
			}
		}
	}// end of loadTrainingData()
	
	private HashMap<Integer, Integer> frequenciesOfBinaryValues(TreeSet<Integer> indicesOfRecords, String colName){
		int indexOfColumn = attributeList.indexOf(colName);
		HashMap<Integer, Integer> frequencyOfBinaryValues = new HashMap<>();
		for(Integer colIndex: indicesOfRecords){
			Record record = records.get(colIndex);
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
	
	@Override
	public String toString() {
		StringBuffer sBuffer = new StringBuffer("");
		sBuffer.append("# Of Attributes: " + this.numberOfAttributes + "\n");
		sBuffer.append("# Of Classes: " + this.numberOfClasses + "\n");
		sBuffer.append("# Of Records: " + this.numberOfRecords + "\n");
		// can add # of nearest neighbors, etc
		for (Record record : this.records) {
			sBuffer.append(record.toString() + "\n");
		}
		sBuffer.deleteCharAt(sBuffer.length() - 1);
		return sBuffer.toString();
	}

}
