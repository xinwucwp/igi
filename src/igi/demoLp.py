import sys
from java.awt import *
from java.lang import *
from java.util import *
from javax.swing import *

from edu.mines.jtk.io import *
from edu.mines.jtk.dsp import *
from edu.mines.jtk.awt import *
from edu.mines.jtk.sgl import *
from edu.mines.jtk.mosaic import *
from edu.mines.jtk.interp import *
from edu.mines.jtk.util.ArrayMath import *


from igi import *

pngDir = None
pngDir = "../../png/"
def main(args):
  goOperInterp()

def goOperInterp():
  t,x,f = getOperAtLogs()
  print min(f)
  print max(f)
  st,sx,s = getSeismicImage()
  lof = LocalOrientFilter(6,3)
  et = lof.applyForTensors(s)   #structure tensors
  et.setEigenvalues(0.0001,1.0) #set anisotropic tensors
  igi = FastImageGuidedInterp(f,t,x) 
  igi.setBiharmonic(0.0)
  igi.setSmoothings(10.0)
  igi.setIters(100,0.001)
  igi.setTensors(et)
  fg = igi.grid(st,sx)
  plot2(f,t,x,s,st,sx,vmin=1.2,vmax=1.9,label="Known value",png="tp2f")
  plot2(f,t,x,s,st,sx,g=fg,vmin=1.2,vmax=1.9,label="Guided harmonic",png="tp2q")

def getOperAtLogs():
  n = 421*2
  t = zerofloat(n)
  x = zerofloat(n)
  f = zerofloat(n)
  oper = zerofloat(n,3)
  amfo = ArrayInputStream("../../data/RCMFOAas001.dat")
  amfo.readFloats(oper)
  amfo.close()
  t = oper[:][0]
  x = oper[:][1]
  f = oper[:][2]
  return t,x,f

def getSeismicImage():
  ft,fx = 0.0,0.0
  dt,dx = 1.0,1.0
  nt,nx = 421,401
  image = zerofloat(nt,nx)
  st,sx = Sampling(nt,dt,ft),Sampling(nx,dx,fx)
  ais = ArrayInputStream("../../data/SeismicDataZeroOffset.dat")
  ais.readFloats(image)
  ais.close()
  return st,sx,image

# plotting
backgroundColor = Color.WHITE
cjet = ColorMap.JET
alpha = fillfloat(1.0,256); alpha[0] = 0.0
ajet = ColorMap.setAlpha(cjet,alpha)

def plot2(f,x1,x2,s,s1,s2,g=None,vmin=1.2,vmax=2.0,
        label=None,png=None,et=None):
  n1 = len(s[0])
  n2 = len(s)
  panel = PlotPanel(1,1,
    PlotPanel.Orientation.X1DOWN_X2RIGHT,
    PlotPanel.AxesPlacement.NONE)
  '''
  panel.setHInterval(2.0)
  panel.setVInterval(0.2)
  panel.setHLabel("Distance (km)")
  panel.setVLabel("Time (s)")
  panel.setHInterval(100.0)
  panel.setVInterval(100.0)
  panel.setHLabel("Pixel")
  panel.setVLabel("Pixel")
  '''
  panel.setVLimits(0,n1-1)
  panel.setHLimits(0,n2-1)
  if label:
    panel.addColorBar(label)
  else:
    panel.addColorBar()
  panel.setColorBarWidthMinimum(180)
  pv = panel.addPixels(s1,s2,s)
  pv.setInterpolation(PixelsView.Interpolation.NEAREST)
  pv.setColorModel(ColorMap.GRAY)
  pv.setClips(-0.2,0.2)
  if g:
    alpha = 0.5
  else:
    g = zerofloat(n1,n2)
    alpha = 0.0
  pv = panel.addPixels(s1,s2,g)
  pv.setInterpolation(PixelsView.Interpolation.NEAREST)
  pv.setColorModel(ColorMap.getJet(alpha))
  pv.setClips(vmin,vmax)
  if et:
    tv = TensorsView(s1,s2,et)
    tv.setOrientation(TensorsView.Orientation.X1DOWN_X2RIGHT)
    tv.setLineColor(Color.YELLOW)
    tv.setLineWidth(3.0)
    tv.setScale(2.0)
    panel.getTile(0,0).addTiledView(tv)
  else:
    cmap = ColorMap(vmin,vmax,ColorMap.JET)
    fs,x1s,x2s = makePointSets(cmap,f,x1,x2)
    for i in range(len(fs)):
      color = cmap.getColor(fs[i][0])
      #color = Color(color.red,color.green,color.blue)
      pv = panel.addPoints(x1s[i],x2s[i])
      pv.setLineStyle(PointsView.Line.NONE)
      pv.setMarkStyle(PointsView.Mark.FILLED_SQUARE)
      pv.setMarkSize(5)
      pv.setMarkColor(color)
  frame = PlotFrame(panel)
  frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE)
  frame.setFontSizeForPrint(8,240)
  #frame.setSize(1240,774)
  #frame.setFontSizeForSlide(1.0,0.8)
  frame.setSize(1290,777)
  frame.setVisible(True)
  if png and pngDir:
    frame.paintToPng(720,3.33,pngDir+"/"+png+".png")
  return frame


def makePointSets(cmap,f,x1,x2):
  sets = {}
  for i in range(len(f)):
    if f[i] in sets:
      points = sets[f[i]]
      points[0].append(f[i])
      points[1].append(x1[i])
      points[2].append(x2[i])
    else:
      points = [[f[i]],[x1[i]],[x2[i]]] # lists of f, x1, x2
      sets[f[i]] = points
  ns = len(sets)
  fs = zerofloat(1,ns)
  x1s = zerofloat(1,ns)
  x2s = zerofloat(1,ns)
  il = 0
  for points in sets:
    fl = sets[points][0]
    x1l = sets[points][1]
    x2l = sets[points][2]
    nl = len(fl)
    fs[il] = zerofloat(nl)
    x1s[il] = zerofloat(nl)
    x2s[il] = zerofloat(nl)
    copy(fl,fs[il])
    copy(x1l,x1s[il])
    copy(x2l,x2s[il])
    il += 1
  return fs,x1s,x2s

#############################################################################
# Run everything on Swing thread.
class RunMain(Runnable):
  def run(self):
    main(sys.argv)
SwingUtilities.invokeLater(RunMain())
