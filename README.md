# spark-model-ryanodine


Using the [2011 sticky cluster model](https://github.com/ananyapavuluri/2011-spark-model) of calcium sparks, stochastic cellular events in cardiomyocytes, this mathematical model simulates repeated calcium sparks over a period of time due to the binding of ryanodine to a receptor.
To mimic this phenomenon, a single RyR in the model is made to be hyperactive by decreasing the channel's mean open time as specified in [Polakova et al., 2015](https://www.ncbi.nlm.nih.gov/pubmed/25772298).
The opening and closing of both free and ryanodine-bound RyRs is modeled stochastically, and a spark is defined by a peak of at least five open RyRs.

This model gives insight into refractory periods between consecutive sparks and spark amplitude restitution. Additionally, by changing parameters in the model, one can also use it to mimic certain drugs, specifically blockers or agonists of β-adrenergic stimulation, thus allowing a thorough analysis of the effects of β-adrenergic stimulatiton on calcium spark amplitude and recovery. [Polakova et al., 2015](https://www.ncbi.nlm.nih.gov/pubmed/25772298) specifies the parameter values for drugs of interest. 

This project aims to use CUDA, a parallel computing platform which performs general purpose computations using the power of a graphics processing unit, to significantly increase simulation speed. The CUDA C++ file (.cu) provided was written and run using Microsoft Visual Studio. It outputs important data in the form of CSV files, which are then read into MATLAB for figure generation and data analysis.

The provided MATLAB scripts generate a histogram of spark to spark delay times and a scatter plot of spark amplitude restitution, respectively.

The program reproduces expected results from [Polakova et al](www.ncbi.nlm.nih.gov/pubmed/2577229) and therefore can be used for further study.

This project was done in the Cardiac Systems Pharmacology Lab of Dr. Eric Sobie, Department of Pharmacological Sciences at the Icahn School of Medicine of Mount Sinai (New York, NY), and with the guidance of M.D./Ph.D. candidate Deanalisa Jones.
