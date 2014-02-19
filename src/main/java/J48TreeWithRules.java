import java.util.List;

import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;


public class J48TreeWithRules {

	public static void main(String[] args) throws Exception
	{
		DataSource trainDataSource = new DataSource("/afs/cs.wisc.edu/u/s/k/skprasad/RA/rule-generator/src/main/resources/data/heart-train.arff");
		Instances data = trainDataSource.getDataSet();
		
		if (data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);
		
		J48 dtree = new J48();
		dtree.buildClassifier(data);
		List<String> rules = dtree.getDecisionTreeRules();
		System.out.println("Rules : " + rules.toString());

	}
}
