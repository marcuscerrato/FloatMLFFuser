package intermidia;

import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;

import org.openimaj.math.statistics.distribution.Histogram;
import org.openimaj.ml.clustering.DoubleCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.DoubleKMeans;
import org.openimaj.util.pair.IntDoublePair;

import com.opencsv.CSVReader;

public class FloatMLFFuser 
{
	private static final boolean msg = false;
	private static int k;
	private static int clusteringSteps;
	private final static int boostFactor = 2;
	
	
	//Usage: <in: first bof file> ... <in: last bof file> <out: fused bof file> <in: k> <in: clustering steps>
	//<in: pooling mode> <in: normalisation> 
	// pooling mode options: 
	// 		kishi inicial options = javg / max / avg / cobmax / cobavg 
	//		harmonic mean = avgharm
	//		geometric mean = avggeom 
	//		lst + <number> 
	//		max + <number>
	// 		minkowski = mnk + <number_1_a_12> 
	//		percentile = per + <number> [ k better ]
	
    public static void main( String[] args ) throws Exception
    {
    	ArrayList<ArrayList<Histogram>> featureArrays = new ArrayList<ArrayList<Histogram>>();   
    	int featureTypes = 0;
    	int featureWords = 0;
    	int shots = 0;
    	
    	/*Set k and maximum clustering steps*/
    	k = Integer.parseInt(args[args.length - 4]);
    	clusteringSteps = Integer.parseInt(args[args.length - 3]);
    	String poolingMode = args[args.length - 2];
    	boolean normalise = args[args.length - 1].equals("normalise");
    	   	
    	//Read different feature word histogram files 
    	for(featureTypes = 0; featureTypes < (args.length - 5); featureTypes++)
    	{
    		CSVReader featureReader = new CSVReader(new FileReader(args[featureTypes]), ' ');
    		String[] line;
    		featureArrays.add(new ArrayList<Histogram>());
    		double maxValue = Double.MIN_VALUE;
    		while ((line = featureReader.readNext()) != null) 
    		{		
    			int fvSize = line.length - 1;    			
    			double fv[] = new double[fvSize];
    			for(int j = 0; j < fvSize; j++)
    			{
    				fv[j] = Double.parseDouble(line[j + 1]);
    				if(fv[j] > maxValue)
    				{
    					maxValue = fv[j];
    				}   
    			}
    			featureArrays.get(featureTypes).add(new Histogram(fv));    		
    		}
     		
    		//Normalize values   		
    		if(normalise)
    		{
    			int size = featureArrays.get(featureTypes).size();
    			int length = 0;
    			for(int i = 0; i < size; i++)
    			{
    				length = featureArrays.get(featureTypes).get(i).length();
    				for(int j = 0; j < length; j++)
    				{
    					double normalisedValue = featureArrays.get(featureTypes).get(i).get(j) / maxValue * 100;
    					featureArrays.get(featureTypes).get(i).setFromDouble(j, normalisedValue);
    				}
    			}
    			if (msg) System.out.print("Type "+featureTypes+"\tNormalized (sizeXlength) = "+(size-1)+" X "+(length-1)+"\t");
    		}

    		//Sums all feature vector lengths for all different types
    		featureWords += featureArrays.get(featureTypes).get(0).length();    		    		
    		featureReader.close();

    		if (msg) System.out.println("Feature words sum = "+featureWords);
    	}    	
    	
    	
    	//Relate each word with its modality
    	int insertedFeatureSum = 0;
    	ArrayList<Integer> wordModality = new ArrayList<Integer>();
    	for(int i = 0; i < featureTypes; i++)
    	{
    		int modalityLength = featureArrays.get(i).get(0).length();
    		for(int j = 0; j < modalityLength; j++)
    		{
    			wordModality.add(i);
    		}
    		insertedFeatureSum += modalityLength;
        	if (msg) System.out.println("Modality with word (typesXlength) = "+ featureTypes +" X "+modalityLength+ " => (sum) "+ insertedFeatureSum);
    	} 	
    	
    	//Compute the transpose matrix
    	shots = featureArrays.get(0).size();
    	double[][] featurePool = new double[featureWords][shots];
    	insertedFeatureSum = 0;    	
    	for(ArrayList<Histogram> arrayList: featureArrays)
    	{
    		int shot = 0;
    		for(Histogram histogram: arrayList)
    		{
    			int lengthHist = histogram.getVector().length;
    			for(int j = 0; j < lengthHist; j++)
    			{
    				featurePool[insertedFeatureSum + j][shot] = histogram.get(j);
    			}
    			shot++;
    		}
    		//Sums already processed feature words
    		insertedFeatureSum += arrayList.get(0).getVector().length;
    	}
    	if (msg) System.out.println("Transpose matrix (wordsxshots) = "+ featureWords +" X "+(shots-1)+ " => (sum) "+ insertedFeatureSum);
    	
    	//Performs clustering of the feature words by their shot histograms
    	DoubleKMeans clusterer = DoubleKMeans.createExact(k, clusteringSteps);
    	DoubleCentroidsResult centroids = clusterer.cluster(featurePool);
    	HardAssigner<double[], double[], IntDoublePair> hardAssigner = centroids.defaultHardAssigner();
    	
    	
    	//Assign each feature word to a cluster that correspond a multimodal feature
    	ArrayList<ArrayList<Integer>> featureGroups = new ArrayList<ArrayList<Integer>>();
    	for(int i = 0; i < k; i++)
    	{
    		featureGroups.add(new ArrayList<Integer>());
    	}
    	
    	for(int featureIndex = 0; featureIndex < featureWords; featureIndex++)
     	{

    		int group = hardAssigner.assign(featurePool[featureIndex]) ;
     		featureGroups.get(group).add(featureIndex);
     	}
 		if (msg) System.out.print(featureGroups.toString() +"\n");
    	
    	int kReal = k; // real size k, when considering a subset of frames
    	
    	//Compute multimodal words histograms by average pooling    	
    	double[][] h;
    	switch(poolingMode)
    	{
    		//Jhuo et al. Average Pooling Strategy 
    		case "javg":
    		{    		
    			h = jhuoAveragePooling(featureGroups, wordModality, featureTypes, featurePool, shots, k);
    			break;
    		}
    		//Standard Max Pooling
    		case "max":
    		{
    			h = maxPooling(featureGroups, featurePool, shots, k, false);
    			break;
    		}
    		//Standard Average Pooling
    		case "avg":
    		{
    			h = averagePooling(featureGroups, featurePool, shots, k, false);
    			break;
    		}
    		//Co-occurrence Boosted Max Pooling
    		case "cobmax":
    		{
    			h = maxPooling(featureGroups, featurePool, shots, k, true);
    			break;
    		}
    		//Co-occurrence Boosted Average Pooling
    		case "cobavg":
    		{
    			h = averagePooling(featureGroups, featurePool, shots, k, true);
    			break;
    		}    			    		    		
    		//Harmonic Average Pooling
    		case "avgharm":
    		{
    			h = averageHarmonicPooling(featureGroups, featurePool, shots, k, false);
    			break;
    		}
    		//Geometric Average Pooling
    		case "avggeom":
    		{
    			h = averageGeometricPooling(featureGroups, featurePool, shots, k, false);
    			break;
    		}
    		default:
    		{
    			String poolingModeType = poolingMode.substring(0, 3);
    			int pollingModeValue = Integer.parseInt(poolingMode.substring(3));
    			
    			if (msg) System.out.println("Pooling mode: "+ poolingModeType + " option "+pollingModeValue);
    			
    			if (poolingModeType.equals("lst")) {
    				int lastFFrames = (pollingModeValue < k) ? pollingModeValue : k;
    				kReal = lastFFrames;
         			if (msg) System.out.println("k real size = "+kReal);
         			h = meanLastFFramesPooling(featureGroups, featurePool, shots, k, false, lastFFrames);
    			}
    			else if (poolingModeType.equals("max")) {
    				int maxNFrames = (pollingModeValue < k) ? pollingModeValue : k;
        			if (msg) System.out.println("N successive frames = "+maxNFrames);
        			h = maxNFramesPooling(featureGroups, featurePool, shots, k, false, maxNFrames);
    			}
    			else if (poolingModeType.equals("mnk")) {
    				int minkExp = (pollingModeValue > 0 && pollingModeValue < 12) ? pollingModeValue : 1;
        			if (msg) System.out.println("Minkowski exponent = "+minkExp);
    				h = averageMinkowskiPooling(featureGroups, featurePool, shots, k, false, minkExp);
    			}
    			else if (poolingModeType.equals("per")) {
    				int perc = (pollingModeValue < 0 || pollingModeValue > 100) ? 100 : pollingModeValue;
        			if (msg) System.out.println("Percent best = "+perc);
    				h = averagePercentilePooling(featureGroups, featurePool, shots, k, false, perc);
    			}
    			else {
    				System.out.println("No pooling mode chosen!");
    				h = zeroPooling(featureGroups, featurePool, shots, k, false);
    			}
    			break;
    		}
    	}
    	
    	
    	//if (msg) System.out.println("Output");
    	//Write multimodal features on output 
    	FileWriter output = new FileWriter(args[args.length - 5]);
    	for(int i = 0; i < shots; i++)
    	{	
    		//Shot number
    		output.write(i + " ");
    		//if (msg) System.out.print(i+"\t");
    		for(int j = 0; j < kReal; j++)
    		{
    			if(j < (kReal - 1))
    			{
    				//if (msg) System.out.print(h[i][j]+" ");
   					output.write(h[i][j] + " ");    					
    			}else
    			{
    				//if (msg) System.out.print(h[i][j]+"\n");
    				output.write(h[i][j] + "\n");    					
    			}
    		} 
    		/* ORIGINAL CODE
    		//Shot number
    		output.write(i + " ");
    		//if (msg) System.out.print(i+"\t");
    		for(int j = 0; j < k; j++)
    		{
    			if(j < (k - 1))
    			{
    				//if (msg) System.out.print(h[i][j]+" ");
   					output.write(h[i][j] + " ");    					
    			}else
    			{
    				//if (msg) System.out.print(h[i][j]+"\n");
    				output.write(h[i][j] + "\n");    					
    			}
    		} 
    		*/   		
    	}    	
    	output.close();
    	if (msg) System.out.println("\nFusion process terminated.");

    }

    private static double[][] jhuoAveragePooling(ArrayList<ArrayList<Integer>> featureGroups, ArrayList<Integer> vectorModality, int vectorModalities,  double featurePool[][], int shots, int k)
    {   	    	
    	double[][] hJhuoAvg = new double[shots][k]; 
    	double[][] doubleHJhuoAvg = new double[shots][k];
    	for(int i = 0; i < shots; i++)
    	{	
    		for(int j = 0; j < k; j++)
    		{    			  			    			
    			/*Divide over different modalities*/    			
    			@SuppressWarnings("unchecked")
    			ArrayList<Integer> words[] = new ArrayList[vectorModalities];
    			for(int l = 0; l < vectorModalities; l++)
    			{
    				words[l] = new ArrayList<Integer>();
    			}
    			for(Integer val: featureGroups.get(j))
    			{
    				words[vectorModality.get(val)].add(val);
    			}  
   			    		
    			/*Sum pairwise individual modalities weights*/
    			double sum = 0;
    			//Iterate each group modalities
    			for(int l = 0; l < vectorModalities; l++)
    			{
    				//Iterate each element of each modality
    				for(int m = 0; m < words[l].size(); m++)
    				{
    					//For each element, combine with other, avoiding repeated combinations
    					for(int n = l + 1; n < vectorModalities; n++)
    					{
    						for(int o = 0; o < words[n].size(); o++)
    						{   							
    							sum += (featurePool[words[l].get(m)][i] + featurePool[words[n].get(o)][i]);
    						}
    					}
    				}
    			} 
    			hJhuoAvg[i][j] = sum / featureGroups.get(j).size();
    			doubleHJhuoAvg[i][j] = Math.ceil(hJhuoAvg[i][j]);
    		}
    	}
    	return doubleHJhuoAvg;
    }

    
    private static double[][] averagePooling(ArrayList<ArrayList<Integer>> featureGroups, double featurePool[][], int shots, int k, boolean boost)
    {
    	double[][] havg = new double[shots][k]; 
    	for(int i = 0; i < shots; i++)
    	{	
    		for(int j = 0; j < k; j++)
    		{
    			if (msg) System.out.format("\n=> i = %d \t j = %d", i, j);
    			//Sum calculation, gather all values of a multimodal word:
    			double sum = 0;
    			for(Integer val: featureGroups.get(j))
    			{
    				sum += featurePool[val][i];    				    			
    				if (msg) System.out.format("\n \t\t val = %d \t featurePool = %f \t sum = %f", val, featurePool[val][i], sum);
    			}
    			//To avoid division by 0 when there are empty clusters.
    			if(featureGroups.get(j).size() > 0)
    			{
    				havg[i][j] = sum / featureGroups.get(j).size();
    				if (msg) System.out.format("\n \t size = %d \t havg = %f", featureGroups.get(j).size(), havg[i][j]);
    			}
    			else
    			{
    				havg[i][j] = sum / 1;
    				if (msg) System.out.format("\n \t size = %d \t havg = %f", featureGroups.get(j).size(), havg[i][j]);
    			}    		
    			
    			//If we want to boost co-occurrences (groups greater than 1)
    			if(featureGroups.get(j).size() > 1 && boost == true)
    			{
    				havg[i][j] *= boostFactor;
    			}
    		}
    	}
    	return havg;
    }
    
    private static double[][] averageHarmonicPooling(ArrayList<ArrayList<Integer>> featureGroups, double featurePool[][], int shots, int k, boolean boost)
    {
    	if (msg) System.out.println("\nHarmonic average");
    	double[][] havgHarm = new double[shots][k]; 
    	for(int i = 0; i < shots; i++)
    	{	
    		for(int j = 0; j < k; j++)
    		{
    			if (msg) System.out.format("\n=> i = %d \t j = %d", i, j);
    			//Sum calculation, gather all values of a multimodal word:
    			double sum = 0.0;
    			int n = featureGroups.get(j).size();
    			for(Integer val: featureGroups.get(j))
    			{
    				if (featurePool[val][i] == 0.0) {
    					sum = 0.0;
    					break;
    				}
    				//double term = (double) (1.0 / featurePool[val][i]); 
    				double term = (double) Math.pow(featurePool[val][i], -1);
   					sum += term;     				    			
    				if (msg) System.out.format("\n \t\t val = %d \t featurePool = %f \t featurePoolInv = %f \t sum = %f", val, featurePool[val][i], term, sum);
    			}
    			//To avoid division by 0 when there are empty clusters.
    			double havgTerm = (double) (0.0);
    			if(n > 0 && sum > 0.0) {
    				havgTerm = (double) Math.pow((sum / n), -1); 
    			}
    			else if (sum == 0.0) { 
    				havgTerm = (double) (0.0);
    			}
    			else {
    				havgTerm = (double) Math.pow((sum), -1);
    			}    		
				havgHarm[i][j] = havgTerm;
    			if (msg) System.out.format("\n \t size = %d \t havg = %f ", n, havgHarm[i][j]);
    			
    			//If we want to boost co-occurrences (groups greater than 1)
    			if(featureGroups.get(j).size() > 1 && boost == true)
    			{
    				havgHarm[i][j] *= boostFactor;
    			}
    		}
    	}
    	return havgHarm;
    }

    private static double[][] averageMinkowskiPooling(ArrayList<ArrayList<Integer>> featureGroups, double featurePool[][], int shots, int k, boolean boost, int minkExp)
    {
    	if (msg) System.out.println("\nMinkowski average");
    	double[][] havgMink = new double[shots][k]; 
    	for(int i = 0; i < shots; i++)
    	{	
    		for(int j = 0; j < k; j++)
    		{	   			
    			if (msg) System.out.format("\n=> i = %d \t j = %d", i, j);
    			
    			//Sum calculation, gather all values of a multimodal word:
    			double sum = 0.0;
    			int n = featureGroups.get(j).size();
    			for(Integer val: featureGroups.get(j))
    			{
    				sum += Math.pow(featurePool[val][i], minkExp);    
    				if (msg) System.out.format("\n \t\t val = %d \t featurePool = %f \t minkExp = %d \t sum = %f", val, featurePool[val][i], minkExp, sum);
    			}
    			//To avoid division by 0 when there are empty clusters.
    			double havgTerm = 0.0;
    			if(n > 0) {
    				havgTerm = sum / n;
    			}
    			else {
    				havgTerm = sum / 1;
    			}    			
    			
    			// root of the number
    			double minkExpInv = (Math.pow(minkExp, -1));
    			havgMink[i][j] = Math.pow(havgTerm, minkExpInv);
    			if (msg) System.out.format("\n \t\t minkExpInv = %f \t havgTerm = %f \t havg = %f", minkExpInv, havgTerm, havgMink[i][j]);
    			
    			//If we want to boost co-occurrences (groups greater than 1)
    			if(featureGroups.get(j).size() > 1 && boost == true)
    			{
    				havgMink[i][j] *= boostFactor;
    			}
    		}
    	}
    	return havgMink;
    }

    private static double[][] averageGeometricPooling(ArrayList<ArrayList<Integer>> featureGroups, double featurePool[][], int shots, int k, boolean boost)
    {
    	if (msg) System.out.println("\nGeometric average");
    	double[][] havgGeom = new double[shots][k]; 
    	for(int i = 0; i < shots; i++)
    	{	
    		for(int j = 0; j < k; j++)
    		{	   	
    			if (msg) System.out.format("\n=> i = %d \t j = %d", i, j);
    			//Sum calculation, gather all values of a multimodal word:
    			double product = 1.0;
    			int n = featureGroups.get(j).size();
    			for(Integer val: featureGroups.get(j))
    			{
    				product *= featurePool[val][i];    				    			
    				if (msg) System.out.format("\n \t\t val = %d \t featurePool = %f \t product = %f", val, featurePool[val][i], product);
    			}
    			
    			// root of the number
    			//double expInv = (Math.pow(10, Math.log10(featureGroups.get(j).size())));
    			
    			if (product > 0.0) havgGeom[i][j] = Math.pow(product, (1/n)); 
    			if (msg) System.out.format("\n \t  havgGeom = %f \t exp = %d ", havgGeom[i][j], n);
    			
    			//If we want to boost co-occurrences (groups greater than 1)
    			if(featureGroups.get(j).size() > 1 && boost == true)
    			{
    				havgGeom[i][j] *= boostFactor;
    			}
    		}
    	}
    	return havgGeom;
    }

    private static double[][] averagePercentilePooling(ArrayList<ArrayList<Integer>> featureGroups, double featurePool[][], int shots, int k, boolean boost, int perc)
    {
    	if (msg) System.out.println("\nPercentile");
    	double[][] havgPerc = new double[shots][k]; 
    	for(int i = 0; i < shots; i++)
    	{	
    		for(int j = 0; j < k; j++)
    		{
    			if (msg) System.out.format("\n=> i = %d \t j = %d", i, j);
    			//Sum calculation, gather all values of a multimodal word:
    			double sum = 0;
    			int n = featureGroups.get(j).size();
    			
    			int quantity = (int) (Math.round((n * perc / 100)) + 1);
    			if (msg) System.out.format("\t n %d \t perc %d \t quantity %d ",n, perc, quantity);
    			Collections.sort(featureGroups.get(j));
    			
    			int count = 0;
    			for(Integer val: featureGroups.get(j))
    			{
    				if (count < (n - quantity) ) {
    					sum += featurePool[val][i];
    				}
    				if (msg) System.out.format("\n \t\t val = %d \t featurePool = %f \t sum = %f \t count = %d x %d - %d \t", val, featurePool[val][i], sum, count, n, quantity);
    				if (msg) System.out.print(count > (n - quantity));
    				count++;
    			}
    			//To avoid division by 0 when there are empty clusters.
    			if(quantity > 0)
    			{
    				havgPerc[i][j] = (double) (sum / quantity);
    			}
    			else
    			{
    				havgPerc[i][j] = (double) (sum / 1);
    			}    		
				if (msg) System.out.format("\n \t size = %d \t havgPerc = %f", quantity, havgPerc[i][j]);
    			
    			//If we want to boost co-occurrences (groups greater than 1)
    			if(featureGroups.get(j).size() > 1 && boost == true)
    			{
    				havgPerc[i][j] *= boostFactor;
    			}
    		}
    	}
    	return havgPerc;
    }
    

    /*
     * Mean value of the last F frames
     */
    private static double[][] meanLastFFramesPooling(ArrayList<ArrayList<Integer>> featureGroups, double featurePool[][], int shots, int k, boolean boost, int lastF)
    {
    	//double[][] hLastF = new double[shots][k];
    	double[][] hLastF = new double[shots][lastF];
    	for(int i = 0; i < shots; i++)
    	{	
    		//for(int j = 0; j < k; j++)
    		int jInicial = k - lastF; 
    		for(int j = jInicial; j < k; j++)
    		{	   			
				int jIndex = j - jInicial;
    			if (msg) System.out.format("\n=> i = %d \t j = %d \t %d ", i, j, jIndex);

				//Sum calculation, gather all values of a multimodal word:
    			double sum = 0;
    			for(Integer val: featureGroups.get(j))
    			{
    				sum += featurePool[val][i]; 
    				if (msg) System.out.format("\n \t\t val = %d \t featurePool = %f \t sum = %f", val, featurePool[val][i], sum);
    			}
    			
    			//To avoid division by 0 when there are empty clusters.
    			if(featureGroups.get(j).size() > 0)
    			{
    				hLastF[i][jIndex] = sum / featureGroups.get(j).size();
    			}
    			else
    			{
    				hLastF[i][jIndex] = sum / 1;
    			} 
				if (msg) System.out.format("\n \t size = %d \t havgLastF = %f", featureGroups.get(j).size(), hLastF[i][jIndex]);
    			   			
    			//hLastF[i][jIndex] = sum / lastF;
				
    			
    			//If we want to boost co-occurrences (groups greater than 1)
    			if(featureGroups.get(j).size() > 1 && boost == true)
    			{
    				hLastF[i][jIndex] *= boostFactor;
    				//hLastF[i][j] *= boostFactor;
    			}
    		}
    	}
    	return hLastF;
    }    

    private static double[][] maxPooling(ArrayList<ArrayList<Integer>> featureGroups, double featurePool[][], int shots, int k, boolean boost)
    {
    	double[][] hmax = new double[shots][k]; 
    	for(int i = 0; i < shots; i++)
    	{	
    		for(int j = 0; j < k; j++)
    		{	
    			//Select maximum value from a multimodal word ocurrence
    			double max = -1;
    			for(Integer val: featureGroups.get(j))
    			{
    				if(featurePool[val][i] > max)
    				{
    					max = featurePool[val][i];
    				}
    			}
    			hmax[i][j] = max;
    			
    			//If we want to boost co-occurrences (groups greater than 1)
    			if(featureGroups.get(j).size() > 1 && boost == true)
    			{
    				hmax[i][j] *= boostFactor;
    			}
    		}
    	}
    	return hmax;
    }    

    /*
     * Max of N successive frames
     */
    private static double[][] maxNFramesPooling(ArrayList<ArrayList<Integer>> featureGroups, double featurePool[][], int shots, int k, boolean boost, int maxN)
    {
    	double[][] hmaxN = new double[shots][k]; 
    	for(int i = 0; i < shots; i++)
    	{	
    		for(int j = 0; j < k; j++)
    		{	 
    			//Select maximum value from a multimodal word ocurrence
    			double max = -1;
    			for(int n = 0; n < maxN; n++) {
    				double sumLocal = 0;
    				double maxLocal = 0;
    				int jLocal = (j+n < k) ? (j+n) : j;
	    			
    				for(Integer val: featureGroups.get(jLocal))
	    			{
	    				sumLocal = sumLocal + featurePool[val][i];
	    			}
	    			maxLocal = sumLocal / maxN;
	    			if(maxLocal > max) max = maxLocal;
    			}
    			hmaxN[i][j] = max;
    		
    			//If we want to boost co-occurrences (groups greater than 1)
    			if(featureGroups.get(j).size() > 1 && boost == true)
    			{
    				hmaxN[i][j] *= boostFactor;
    			}
    			/* ORIGINAL
    			//Select maximum value from a multimodal word ocurrence
    			double max = -1;
    			for(Integer val: featureGroups.get(j))
    			{
    				if(featurePool[val][i] > max)
    				{
    					max = featurePool[val][i];
    				}
    			}
    			hmaxN[i][j] = max;
    			
    			//If we want to boost co-occurrences (groups greater than 1)
    			if(featureGroups.get(j).size() > 1 && boost == true)
    			{
    				hmaxN[i][j] *= boostFactor;
    			}
    			*/
    		}
    	}
    	return hmaxN;
    }    

    private static double[][] zeroPooling(ArrayList<ArrayList<Integer>> featureGroups, double featurePool[][], int shots, int k, boolean boost)
    {
    	if (msg) System.out.println("\nZero");
    	double[][] h = new double[shots][k]; 
    	for(int i = 0; i < shots; i++)
    	{	
    		for(int j = 0; j < k; j++)
    		{
    			if (msg) System.out.format("\n=> i = %d \t j = %d", i, j);
    			h[i][j] = 0.0;
    		}
    	}
    	return h;
    }
    

}