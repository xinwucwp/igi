"""
Demo of 2d interpolation
Author: Xinming Wu, University of Texas at Austin
Version: 2018.03.01
"""

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
  #goDemoNotreDame() 
  goDemoTeapot()

#a demo of 2D isotropic interpolation
def goDemoNotreDame():
  x,y,f = getDataNotreDame()
  sx,sy = getSamplingsNotreDame(grid="fine")
  fx = putDataOnGrid(f,x,y,sx,sy)
  igi = FastImageGuidedInterp(f,x,y) 
  igi.setBiharmonic(100)
  igi.setSmoothings(10.0)
  igi.setIters(100,0.001)
  g = igi.grid(sx,sy)
  plot2(f,x,y,fx,sx,sy,title="notreDame known points",
          contours=False,points=False, png="notreDameKnownPoints")
  plot2(f,x,y,g,sx,sy,title="notreDame interpolation",png="notreDameInterp")
  plotSurface(f,x,y,g,sx,sy,png="notreDameSurface")

def goDemoTeapot():
  t,x,f = getDataTeapot()
  st,sx,s = getImageTeapot()
  lof = LocalOrientFilter(6,3)
  et = lof.applyForTensors(s) # structure tensors
  et.setEigenvalues(0.01,1.0)
  igi = FastImageGuidedInterp(f,t,x,None) 
  igi.setBiharmonic(100)
  igi.setSmoothings(10.0)
  igi.setIters(200,0.001)
  igi.setTensors(et)
  fg = igi.grid(st,sx)
  plot2x(f,t,x,s,st,sx,vmin=0.0,vmax=1,label="Known value",png="tp2f")
  plot2x(f,t,x,s,st,sx,g=fg,vmin=0.0,vmax=1,label="Guided harmonic",png="tp2q")


def getDataTeapot():
  """
  Values set interactively by Dave Hale.
  """
  txf = [
    30, 69,0.50,  99, 72,0.50, 153, 69,0.50, 198, 68,0.50, 
    63, 71,0.90, 128, 72,0.90, 176, 69,0.90,
    29,172,0.35,  97,173,0.35, 150,173,0.35, 192,176,0.35,
    63,173,0.75, 127,174,0.75, 172,174,0.75,
    33,272,0.20, 103,270,0.20, 160,267,0.20, 199,267,0.20,
    70,271,0.60, 134,268,0.60, 179,267,0.60]
  n = len(txf)/3
  t = zerofloat(n)
  x = zerofloat(n)
  f = zerofloat(n)
  copy(n,0,3,txf,0,1,t)
  copy(n,1,3,txf,0,1,x)
  copy(n,2,3,txf,0,1,f)
  #t = add(0.5,mul(0.004,t))
  #x = add(0.0,mul(0.025,x))
  return t,x,f

def getImageTeapot():
  #ft,fx = 0.500,0.000
  #dt,dx = 0.004,0.025
  ft,fx = 0.0,0.0
  dt,dx = 1.0,1.0
  nt,nx = 251,357
  image = zerofloat(nt,nx)
  st,sx = Sampling(nt,dt,ft),Sampling(nx,dx,fx)
  ais = ArrayInputStream("../../data/tp73.dat")
  ais.readFloats(image)
  ais.close()
  return st,sx,image

def makeImageTensors(s):
  """ 
  Returns tensors for guiding along features in specified image.
  """
  sigma = 3
  n1,n2 = len(s[0]),len(s)
  lof = LocalOrientFilter(sigma)
  t = lof.applyForTensors(s) # structure tensors
  c = coherence(sigma,t,s) # structure-oriented coherence c
  c = clip(0.0,0.99,c) # c clipped to range [0,1)
  t.scale(sub(1.0,c)) # scale structure tensors by 1-c
  t.invertStructure(1.0,1.0) # invert and normalize
  return t

def coherence(sigma,t,s):
  lsf = LocalSemblanceFilter(sigma,4*sigma)
  return lsf.semblance(LocalSemblanceFilter.Direction2.V,t,s)


def putDataOnGrid(f,x,y,sx,sy):
  """ facilitates comparison of different gridders """
  n = len(f)
  ny,nx = sy.count,sx.count
  fx = zerofloat(ny,nx)
  for i in range(n):
    ix = sx.indexOfNearest(x[i])
    iy = sy.indexOfNearest(y[i])
    x[i] = sx.getValue(ix)
    y[i] = sy.getValue(iy)
    fx[iy][ix] = f[i]
    fx[iy+1][ix] = f[i]
    fx[iy-1][ix] = f[i]
    fx[iy][ix-1] = f[i]
    fx[iy][ix+1] = f[i]
    fx[iy-1][ix-1] = f[i]
    fx[iy-1][ix+1] = f[i]
    fx[iy+1][ix-1] = f[i]
    fx[iy+1][ix+1] = f[i]
  return fx

def getSamplingsNotreDame(grid="fine"):
  fx,fy = -5.0,-5.0
  dx,dy = 2.00,2.00; nx,ny = 165,165
  if grid=="coarser": dx,dy = 8.00,8.00; nx,ny = 42,42
  elif grid=="coarse": dx,dy = 4.00,4.00; nx,ny = 83,83
  elif grid=="medium": dx,dy = 2.00,2.00; nx,ny = 165,165
  elif grid=="fine": dx,dy = 1.00,1.00; nx,ny = 329,329
  elif grid=="finer": dx,dy = 0.50,0.50; nx,ny = 657,657
  elif grid=="finest": dx,dy = 0.25,0.25; nx,ny = 1313,1313
  sx,sy = Sampling(nx,dx,fx),Sampling(ny,dy,fy)
  return sx,sy

def getDataNotreDame():
  """
  Elevation data from Davis, J.C., 2002, Statistics and Data Analysis in
  Geology, 3rd Edition: Wiley, page 374. The name of the file from which 
  these data were copied is NOTREDAM.TXT. These data were collected over
  a small area by students in a surveying class.
  These data are used in Figure 3 of Wessel, P., 2009, A general-purpose 
  Green's function-based interpolator: Computers and Geosciences 35, 
  1247-1254.
  This function returns three float arrays x, y, z containing Eastings,
  Northings and elevations. Units are unknown. Davis writes that one 
  map unit = 50 ft, but this implies extremely steep terrain, with 
  slopes exceeding 45 degrees.
  """
  xyzNotreDame = [
  0.3, 6.1, 870.0, 1.4, 6.2, 793.0, 2.4, 6.1, 755.0, 3.6, 6.2, 690.0,
  5.7, 6.2, 800.0, 1.6, 5.2, 800.0, 2.9, 5.1, 730.0, 3.4, 5.3, 728.0,
  3.4, 5.7, 710.0, 4.8, 5.6, 780.0, 5.3, 5.0, 804.0, 6.2, 5.2, 855.0,
  0.2, 4.3, 830.0, 0.9, 4.2, 813.0, 2.3, 4.8, 762.0, 2.5, 4.5, 765.0,
  3.0, 4.5, 740.0, 3.5, 4.5, 765.0, 4.1, 4.6, 760.0, 4.9, 4.2, 790.0,
  6.3, 4.3, 820.0, 0.9, 3.2, 855.0, 1.7, 3.8, 812.0, 2.4, 3.8, 773.0,
  3.7, 3.5, 812.0, 4.5, 3.2, 827.0, 5.2, 3.2, 805.0, 6.3, 3.4, 840.0,
  0.3, 2.4, 890.0, 2.0, 2.7, 820.0, 3.8, 2.3, 873.0, 6.3, 2.2, 875.0,
  0.6, 1.7, 873.0, 1.5, 1.8, 865.0, 2.1, 1.8, 841.0, 2.1, 1.1, 862.0,
  3.1, 1.1, 908.0, 4.5, 1.8, 855.0, 5.5, 1.7, 850.0, 5.7, 1.0, 882.0,
  6.2, 1.0, 910.0, 0.4, 0.5, 940.0, 1.4, 0.6, 915.0, 1.4, 0.1, 890.0,
  2.1, 0.7, 880.0, 2.3, 0.3, 870.0, 3.1, 0.0, 880.0, 4.1, 0.8, 960.0,
  5.4, 0.4, 890.0, 6.0, 0.1, 860.0, 5.7, 3.0, 830.0, 3.6, 6.0, 705.0]
  nNotreDame = len(xyzNotreDame)/3
  xNotreDame = zerofloat(nNotreDame)
  yNotreDame = zerofloat(nNotreDame)
  zNotreDame = zerofloat(nNotreDame)
  copy(nNotreDame,0,3,xyzNotreDame,0,1,xNotreDame)
  copy(nNotreDame,1,3,xyzNotreDame,0,1,yNotreDame)
  copy(nNotreDame,2,3,xyzNotreDame,0,1,zNotreDame)
  xNotreDame = mul(50.0,xNotreDame) # Davis says one map unit = 50 ft,
  yNotreDame = mul(50.0,yNotreDame) # but this makes the terrain steep!
  return xNotreDame,yNotreDame,zNotreDame

#############################################################################
# plotting
backgroundColor = Color.WHITE
cjet = ColorMap.JET
alpha = fillfloat(1.0,256); alpha[0] = 0.0
ajet = ColorMap.setAlpha(cjet,alpha)


def plot2(f,x1,x2,g,s1,s2,title=None,png=None,contours=True,points=True):
  n1 = len(g[0])
  n2 = len(g)
  panel = panel2()
  panel.setHLimits(-5.0,323.0)
  panel.setVLimits(-5.0,323.0)
  panel.setHInterval(100.0)
  panel.setVInterval(100.0)
  panel.setHLabel("Easting (ft)")
  panel.setVLabel("Northing (ft)")
  panel.addColorBar("Elevation (ft)")
  panel.setColorBarWidthMinimum(65)
  pv = panel.addPixels(s1,s2,g)
  pv.setClips(min(f),max(f))
  pv.setInterpolation(PixelsView.Interpolation.NEAREST)
  if points:
    pv.setColorModel(cjet)
  else:
    pv.setColorModel(ajet)
  if contours:
    cv = panel.addContours(s1,s2,g)
    cv.setContours(Sampling(10,25.0,700.0)) # 700 - 925
    cv.setLineColor(Color.BLACK)
  if points:
    pv = panel.addPoints(x1,x2)
    pv.setLineStyle(PointsView.Line.NONE)
    pv.setMarkStyle(PointsView.Mark.FILLED_CIRCLE)
    pv.setMarkSize(6)
  frame2(panel,title,png)

def panel2():
  panel = PlotPanel(1,1,
    PlotPanel.Orientation.X1RIGHT_X2UP,
    PlotPanel.AxesPlacement.LEFT_BOTTOM)
  return panel

def frame2(panel,title=None,png=None):
  frame = PlotFrame(panel)
  frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE)
  if title:
    panel.setTitle(title)
    frame.setFontSize(16)
    frame.setSize(1100,960)
  else:
    frame.setFontSize(16)
    frame.setSize(1170,897)
  frame.setVisible(True)
  if png and pngDir:
    frame.paintToPng(720,3.33,pngDir+"/"+png+".png")
  return frame

def plotSurface(f,x1,x2,g,s1,s2,png=None):
  g = transpose(g)
  n = len(f)
  #f = mul(0.5,f)
  #g = mul(0.5,g)
  xyz = zerofloat(3*n)
  copy(n,0,1,x1,0,3,xyz)
  copy(n,0,1,x2,1,3,xyz)
  copy(n,0,1, f,2,3,xyz)
  tg = TriangleGroup(True,s1,s2,g)
  pg = PointGroup(s1.delta*s1.count/100.0,xyz);
  pg.setStates(StateSet.forTwoSidedShinySurface(Color.RED))
  sf = SimpleFrame()
  sf.orbitView.setScale(1.2)
  sf.orbitView.setAxesOrientation(AxesOrientation.XOUT_YRIGHT_ZUP)
  sf.setWorldSphere(s1.first,s2.first,min(f),s1.last,s2.last,max(f))
  sf.world.addChild(tg)
  sf.world.addChild(pg)
  sf.setSize(1000,1000)
  if png and pngDir:
    sf.paintToFile(pngDir+png+".png")

def plot2x(f,x1,x2,s,s1,s2,g=None,vmin=0.0,vmax=1.0,
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
  pv.setInterpolation(PixelsView.Interpolation.LINEAR)
  pv.setColorModel(ColorMap.GRAY)
  pv.setClips(-4.5,4.5)
  if g:
    alpha = 0.5
  else:
    g = zerofloat(n1,n2)
    alpha = 0.0
  pv = panel.addPixels(s1,s2,g)
  pv.setInterpolation(PixelsView.Interpolation.LINEAR)
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
