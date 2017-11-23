package intermidia;

import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;

import org.openimaj.math.statistics.distribution.Histogram;
import org.openimaj.ml.clustering.DoubleCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.DoubleKMeans;
import org.openimaj.util.pair.IntDoublePair;

import com.opencsv.CSVReader;

public class FloatMLFFuser 
{
	private static int k;
	private static int clusteringSteps;
	private final static int boostFactor = 2;
	
	
	//Usage: <in: first bof file> ... <in: last bof file> <out: fused bof file> <in: k> <in: clustering steps>
	//<in: pooling mode> <in: normalisation> 
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
    			for(int i = 0; i < featureArrays.get(featureTypes).size(); i++)
    			{
    				
    				for(int j = 0; j < featureArrays.get(featureTypes).get(i).length(); j++)
    				{
    					double normalisedValue = featureArrays.get(featureTypes).get(i).get(j) / maxValue * 100;
    					featureArrays.get(featureTypes).get(i).setFromDouble(j, normalisedValue);
    				}
    			}
    		}
    		
    			

    		//Sums all feature vector lengths for all different types
    		featureWords += featureArrays.get(featureTypes).get(0).length();    		    		
    		featureReader.close();
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
    			for(int j = 0; j < histogram.getVector().length; j++)
    			{
    				featurePool[insertedFeatureSum + j][shot] = histogram.get(j);
    			}
    			shot++;
    		}
    		//Sums already processed feature words
    		insertedFeatureSum += arrayList.get(0).getVector().length;
    	}
    	
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
    		default:
    		{
    			h = averagePooling(featureGroups, featurePool, shots, k, false);
    			break;
    		}
    	}
    	
    	//Write multimodal features on output 
    	FileWriter output = new FileWriter(args[args.length - 5]);
    	for(int i = 0; i < shots; i++)
    	{	
    		//Shot number
    		output.write(i + " ");
    		for(int j = 0; j < k; j++)
    		{
    			if(j < (k - 1))
    			{
   					output.write(h[i][j] + " ");    					
    			}else
    			{
    				output.write(h[i][j] + "\n");    					
    			}
    		}    		
    	}    	
    	output.close();
    	//System.out.println("Fusion process terminated.");
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
    			//Sum calculation, gather all values of a multimodal word:
    			double sum = 0;
    			for(Integer val: featureGroups.get(j))
    			{
    				sum += featurePool[val][i];    				    			
    			}
    			//To avoid division by 0 when there are empty clusters.
    			if(featureGroups.get(j).size() > 0)
    			{
    				havg[i][j] = sum / featureGroups.get(j).size();
    			}
    			else
    			{
    				havg[i][j] = sum / 1;
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

}