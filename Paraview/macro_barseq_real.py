# trace generated using paraview version 6.1.0-RC1
#import paraview

#### import the simple module from the paraview
from paraview.simple import *
import paraview.simple as pvs
pvs._DisableFirstRenderCameraReset()


# get active source.
source = GetActiveSource()
view = GetActiveViewOrCreate("RenderView")


display = GetRepresentation(source, view=view)

display.SetRepresentationType('Point Gaussian')


# rescale color and/or opacity maps used to exactly fit the current data range
display.RescaleTransferFunctionToDataRange(False, False)

# Properties modified 
display.GaussianRadius = 1.9

# Properties modified 
display.ScaleByArray = 1

# Properties modified
display.SetScaleArray = ['POINTS', 'wi']

display.UseScaleFunction = False
