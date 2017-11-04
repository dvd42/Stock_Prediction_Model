import regression as r
import handler as h

# Initialization
verbose,scale,variations,custom,alpha,epsilon,max_iter,data = h.process_runtime_arguments()
path = h.create_dir(scale,verbose,custom)
h.add_file_header(scale,variations,path)


print "Running model with:"
print "Scale: %d " % scale
print "Variations: %d" % variations


# Processing data from dataset
X = data.iloc[1:, 3:].values.astype('float64')
X = (X+1) * scale
Y = data.iloc[1:, 2].values.astype('float64')
Y = (Y+1) * scale
tags = data.iloc[0,3:].values



# Get errors for each model and variation
for i in range(1,5):
        
    ratio = 0.9 - i/float(10)
    r.store_mean_error(ratio,i,tags,verbose,variations,scale,X,Y,path,custom,alpha,epsilon,max_iter)
    
    
h.add_file_footer(path)
    
    
    