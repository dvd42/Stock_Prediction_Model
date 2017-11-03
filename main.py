import regression as r
import handler as h

# Initialization
verbose,scale,variations,data = h.process_runtime_arguments()


h.create_dir(scale,verbose)

h.add_file_header(scale,variations)

print "Running model with:"
print "Scale: %d " % scale
print "Variations: %d" % variations


#Processing data from dataset
X = data.iloc[1:, 3:].values.astype('float64')
X = (X+1) * scale
Y = data.iloc[1:, 2].values.astype('float64')
Y = (Y+1) * scale
tags = data.iloc[0,3:].values



#We generate evaluation and training set randomly
for i in range(1,5):
        
    ratio = 0.9 - i/float(10)
    r.store_mean_error(ratio,i,tags,verbose,variations,scale,X,Y)
    
    
h.add_file_footer()
    
    
    