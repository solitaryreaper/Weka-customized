import weka.classifiers.rules.PART;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class WekaCustomizationTest {

	public static void main(String[] args) throws Exception
	{
		//testPART();
		//testJ48();
		testRandomForest();
	}
	
	private static void testPART() throws Exception
	{
		Instances data = getInstances();
		PART dlist = new PART();
		dlist.buildClassifier(data);
		System.out.println("\n\nPART rules ..");
		for(String rule : dlist.getDecisionListRules()) {
			System.out.println(rule.toString());
		}
	}
	
	private static void testJ48() throws Exception
	{
		Instances data = getInstances();
		J48 dtree = new J48();
		dtree.buildClassifier(data);
		System.out.println("\n\nJ48 rules ..");
		for(String rule : dtree.getDecisionTreeRules()) {
			System.out.println(rule);
		}
	}
	
	private static void testRandomForest() throws Exception
	{
		Instances data = getInstances();
		RandomForest dtree = new RandomForest();
		dtree.buildClassifier(data);
		/*
		System.out.println("\n\nRandom Forest rules : ");
		for(String rule : dtree.getRandomForestRules()) {
			System.out.println(rule);
		}
		*/
		
		for(Instance instance : data) {
			double entropy = dtree.getVotingEntropyForInstance(instance);
			if(Double.compare(entropy, 0.0) != 0) {
				System.out.println("Entropy : " + entropy);				
			}
		}
		
	}
	
	private static Instances getInstances() throws Exception
	{
		DataSource trainDataSource = new DataSource("/afs/cs.wisc.edu/u/s/k/skprasad/RA/rule-generator/src/main/resources/data/Restaurant.arff");
		Instances data = trainDataSource.getDataSet();
		
		if (data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);
		
		return data;
	}
}
