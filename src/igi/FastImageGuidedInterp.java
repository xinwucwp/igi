package igi;

import edu.mines.jtk.dsp.*;
import edu.mines.jtk.util.*;
import static edu.mines.jtk.util.ArrayMath.*;

/**
 * Fast image-guided interpolation across faults by solving 
 * weighted anisotropic harmonic (Gaussian operator) 
 * and bi-harmonic (Laplacian operator) interpolation.
 * @author Xinming Wu
 * @version 2015.06.07
 */
public class FastImageGuidedInterp {

  /**
   * Constructs an interpolator.
   * @param fx know values at the known points.
   * @param x1 1st coordinates of known points.
   * @param x2 2nd coordinates of known points.
   */
  public FastImageGuidedInterp(
    float[] fx, float[] x1, float[] x2) 
  {
    _fx = copy(fx);
    _x1 = copy(x1);
    _x2 = copy(x2);
  }

  /**
   * Constructs an interpolator.
   * @param fx know values at the known points.
   * @param x1 1st coordinates of known points.
   * @param x2 2nd coordinates of known points.
   * @param x3 3rd coordinates of known points.
   */
  public FastImageGuidedInterp(
    float[] fx, float[] x1, float[] x2, float[] x3) 
  {
    _fx = copy(fx);
    _x1 = copy(x1);
    _x2 = copy(x2);
    if(x3!=null){_x3=copy(x3);}
  }

  /**
   * Set smoothing for preconditioning in a CG solver.
   * @param sigma1 smoother half-width for 1st dimension.
   * @param sigma2 smoother half-width for 2nd dimension.
   */
  public void setSmoothings(double sigma1, double sigma2) {
    _sigma1 = (float)sigma1;
    _sigma2 = (float)sigma2;
  }

  /**
   * Set iterations for a CG solver.
   * @param niter number of the maximum iterations.
   */
  public void setIters(int niter) {
    _niter =  niter;
  }

  /**
   * Set 2D structure tensors for 2D image-guided interpolation.
   * @param d2 2D structure tensor field.
   */
  public void setTensors(Tensors2 d2) {
    _d2 = d2;
  }

  /**
   * Set 3D structure tensors for 3D image-guided interpolation.
   * @param d3 3D structure tensor field.
   */
  public void setTensors(Tensors3 d3) {
    _d3 = d3;
  }

  /**
   * Apply for 2D image-guided interpolation.
   * @param s1 sampling in the 1st (vertical) dimension.
   * @param s2 sampling in the 2nd (lateral) dimension.
   * @return array of the interpolated image.
   */
  public float[][] grid(Sampling s1, Sampling s2) {
    int n1 = s1.getCount();
    int n2 = s2.getCount();
    float[][] r = new float[n2][n1];
    float[][] wp = fillfloat(1f,n1,n2);
    int np = _x1.length;
    for (int ip=0; ip<np; ++ip) {
      _x1[ip] = (float)s1.indexOfNearest(_x1[ip]);
      _x2[ip] = (float)s2.indexOfNearest(_x2[ip]);
    }
    float[][] b = new float[n2][n1];
    setInitial(r);
    VecArrayFloat2 vb = new VecArrayFloat2(b);
    VecArrayFloat2 vr = new VecArrayFloat2(r);
    Smoother2 sm2 = new Smoother2(_sigma1,_sigma2,wp);
    CgSolver cg = new CgSolver(_small,_niter);
    A2 a2 = new A2(_d2,wp);
    M2 m2 = new M2(_x1,_x2,sm2);
    vb.zero();
    cg.solve(a2,m2,vb,vr);
    return r;
  }

  /**
   * Apply for 2D image-guided and weighted interpolation.
   * @param s1 sampling in the 1st (vertical) dimension.
   * @param s2 sampling in the 2nd (lateral) dimension.
   * @param wp weights, low values near faults, high values elsewhere.
   * @return array of the interpolated image.
   */
  public float[][] grid(Sampling s1, Sampling s2, float[][] wp) {
    int n1 = s1.getCount();
    int n2 = s2.getCount();
    float[][] r = new float[n2][n1];
    int np = _x1.length;
    for (int ip=0; ip<np; ++ip) {
      _x1[ip] = (float)s1.indexOfNearest(_x1[ip]);
      _x2[ip] = (float)s2.indexOfNearest(_x2[ip]);
    }
    float[][] b = new float[n2][n1];
    setInitial(r);
    VecArrayFloat2 vb = new VecArrayFloat2(b);
    VecArrayFloat2 vr = new VecArrayFloat2(r);
    Smoother2 sm2 = new Smoother2(_sigma1,_sigma2,null);
    CgSolver cg = new CgSolver(_small,_niter);
    A2 a2 = new A2(_d2,wp);
    M2 m2 = new M2(_x1,_x2,sm2);
    vb.zero();
    cg.solve(a2,m2,vb,vr);
    return r;
  }


  /**
   * Apply for 3D image-guided interpolation.
   * @param sp screen points on faults.
   * @param wp weights, zeros on faults, ones elsewhere.
   * @return array of the interpolated image.
   */
  public float[][][] apply(
    float[][][] sp, float[][][] wp) {
    int n3 = wp.length;
    int n2 = wp[0].length;
    int n1 = wp[0][0].length;
    float[][][] b = new float[n3][n2][n1];
    float[][][] r = new float[n3][n2][n1];
    setInitial(r);
    VecArrayFloat3 vr = new VecArrayFloat3(r);
    VecArrayFloat3 vb = new VecArrayFloat3(b);
    Smoother3 s3 = new Smoother3(_sigma1,_sigma2,_sigma2,wp);
    CgSolver cg = new CgSolver(_small,_niter);
    A3 a3 = new A3(_d3,sp,wp);
    M3 m3 = new M3(_x1,_x2,_x3,s3);
    vb.zero();
    cg.solve(a3,m3,vb,vr);
    return r;
  }

  private void setInitial(float[][] x) {
    if(_x1==null||_x2==null||_fx==null){return;}
    int np = _x1.length;
    for (int ip=0; ip<np; ++ip) {
      int i1 = (int)_x1[ip];
      int i2 = (int)_x2[ip];
      x[i2][i1] = _fx[ip];
    }
  }

  private void setInitial(float[][][] x) {
    if(_x1==null||_x2==null||_x3==null||_fx==null){return;}
    int np = _x1.length;
    for (int ip=0; ip<np; ++ip) {
      int i1 = (int)_x1[ip];
      int i2 = (int)_x2[ip];
      int i3 = (int)_x3[ip];
      x[i3][i2][i1] = _fx[ip];
    }
  }

  ///////////////////////////////////////////////////////////////////////////
  // private
  private Tensors2 _d2;
  private Tensors3 _d3;

  private static float[] _x1 = null; // 1st coordinates of the known points
  private static float[] _x2 = null; // 2nd coordinates of the known points
  private static float[] _x3 = null; // 3rd coordinates of the known points
  private static float[] _fx = null; // known values at the known points
  private float _sigma1 = 10.0f; // half-width of smoother in 1st dimension
  private float _sigma2 = 10.0f; // half-width of smoother in 2nd dimension
  private float _small = 0.010f; // stop CG iterations if residuals are small
  private int _niter = 800; // maximum number of inner CG iterations

  private static class A2 implements CgSolver.A {
    A2(Tensors2 et, float[][] wp) 
    {
      _et = et;
      _wp = wp;
    }
    public void apply(Vec vx, Vec vy) {
      VecArrayFloat2 v2x = (VecArrayFloat2)vx;
      VecArrayFloat2 v2y = (VecArrayFloat2)vy;
      float[][] x = v2x.getArray();
      float[][] y = v2y.getArray();
      float[][] z = copy(x);
      float[][] t = copy(x);
      VecArrayFloat2 v2t = new VecArrayFloat2(t);
      v2y.zero();
      v2t.zero();
      applyLhs(_et,_wp,z,y);
      applyLhs(_et,_wp,y,t);
      v2y.add(1.f,v2t,100f);
    }

    private float[][] _wp=null;
    private Tensors2 _et = null;
  }

  // Preconditioner; includes smoothers and constraints.
  private static class M2 implements CgSolver.A {
    M2(float[] x1, float[] x2, Smoother2 s2) {
      _x1 = x1;
      _x2 = x2;
      _s2 = s2;
    }
    public void apply(Vec vx, Vec vy) {
      VecArrayFloat2 v2x = (VecArrayFloat2)vx;
      VecArrayFloat2 v2y = (VecArrayFloat2)vy;
      float[][] x = v2x.getArray();
      float[][] y = v2y.getArray();
      copy(x,y);
      constrain(_x1,_x2,y);
      _s2.apply(y);
      constrain(_x1,_x2,y);
    }
    private Smoother2 _s2;
    private float[] _x1,_x2;
  }

  private static class A3 implements CgSolver.A {
    A3(Tensors3 et, float[][][] sp, float[][][] wp) 
    {
      _et = et;
      _sp = sp;
      _wp = wp;
    }
    public void apply(Vec vx, Vec vy) {
      VecArrayFloat3 v3x = (VecArrayFloat3)vx;
      VecArrayFloat3 v3y = (VecArrayFloat3)vy;
      float[][][] x = v3x.getArray();
      float[][][] y = v3y.getArray();
      float[][][] z = copy(x);
      float[][][] t = copy(x);
      VecArrayFloat3 v3t = new VecArrayFloat3(t);
      v3y.zero();
      v3t.zero();
      applyLhs(_et,_wp,z,y);
      applyLhs(_et,_wp,y,t);
      v3y.add(1.f,v3t,50f);
      if(_sp!=null) {
        screenLhs(_sp[0],_sp[1],_sp[3][0],z,y);
      }
    }

    private Tensors3 _et = null;
    private float[][][] _wp=null;
    private float[][][] _sp=null;
  }

  // Preconditioner; includes smoothers and constraints.
  private static class M3 implements CgSolver.A {
    M3(float[] x1, float[] x2, float[] x3, Smoother3 s3) {
      _x1 = x1;
      _x2 = x2;
      _x3 = x3;
      _s3 = s3;
    }
    public void apply(Vec vx, Vec vy) {
      VecArrayFloat3 v3x = (VecArrayFloat3)vx;
      VecArrayFloat3 v3y = (VecArrayFloat3)vy;
      float[][][] x = v3x.getArray();
      float[][][] y = v3y.getArray();
      copy(x,y);
      constrain(_x1,_x2,_x3,y);
      _s3.apply(y);
      constrain(_x1,_x2,_x3,y);
    }
    private Smoother3 _s3;
    private float[] _x1,_x2,_x3;
  }

  private static void constrain(
    float[] x1, float[] x2, float[][] x) 
  {
    if (x1!=null && x2!=null) {
      int np = x1.length;
      for (int ip=0; ip<np; ++ip) {
        int i1 = (int)x1[ip]; 
        int i2 = (int)x2[ip]; 
        x[i2][i1] = 0.0f;
      }
    }
  }

  private static void constrain(
    float[] x1, float[] x2, float[] x3, float[][][] x) 
  {
    if (x1!=null && x2!=null && x3!=null) {
      int np = x1.length;
      for (int ip=0; ip<np; ++ip) {
        int i1 = (int)x1[ip]; 
        int i2 = (int)x2[ip]; 
        int i3 = (int)x3[ip]; 
        x[i3][i2][i1] = 0.0f;
      }
    }
  }


  private void makeRhs(float[][] r) {
    int np = _x1.length;
    for (int ip=0; ip<np; ++ip) {
      int i1 = (int)_x1[ip];
      int i2 = (int)_x2[ip];
      r[i2][i1] = _fx[ip];
    }

  }

  private static void applyLhs(
    final Tensors2 d, final float[][] wp, 
    final float[][] x, final float[][] y)
  {
    zero(y);
    int n2 = x.length;
    int n1 = x[0].length;
    float[] ds = fillfloat(1.0f,3);
    ds[0] = 1.0f;
    ds[1] = 0.0f;
    ds[2] = 1.0f;
    for (int i2=1; i2<n2; ++i2) {
      for (int i1=1; i1<n1; ++i1) {
        if(d!=null){d.getTensor(i1,i2,ds);}
        float wpi = (wp!=null)?wp[i2][i1]:1.0f;
        float wps = wpi*wpi;
        float d11 = ds[0]*wps;
        float d12 = ds[1]*wps;
        float d22 = ds[2]*wps;
        float xa = 0.0f;
        float xb = 0.0f;
        xa += x[i2  ][i1  ];
        xb -= x[i2  ][i1-1];
        xb += x[i2-1][i1  ];
        xa -= x[i2-1][i1-1];
        float x1 = 0.5f*(xa+xb);
        float x2 = 0.5f*(xa-xb);
        float y1 = d11*x1+d12*x2;
        float y2 = d12*x1+d22*x2;
        float ya = 0.5f*(y1+y2);
        float yb = 0.5f*(y1-y2);
        y[i2  ][i1  ] += ya;
        y[i2  ][i1-1] -= yb;
        y[i2-1][i1  ] += yb;
        y[i2-1][i1-1] -= ya;
      }
    }

  }

  private static void applyLhs(
    final Tensors3 d, final float[][][] wp, 
    final float[][][] x, final float[][][] y)
  { 
    final int n3 = y.length;
    Parallel.loop(1,n3,2,new Parallel.LoopInt() {
    public void compute(int i3) {
      applyLhsSlice3(i3,d,wp,x,y);
    }});
    Parallel.loop(2,n3,2,new Parallel.LoopInt() {
    public void compute(int i3) {
      applyLhsSlice3(i3,d,wp,x,y);
    }});
  }

  private static void screenLhs(
    float[][] cp, float[][] cm, float[] fl, float[][][] x, float[][][] y) 
  {
    int nc = cp[0].length;
    for (int ic=0; ic<nc; ++ic) {
      int i1p = (int)cp[0][ic];
      int i2p = (int)cp[1][ic];
      int i3p = (int)cp[2][ic];
      int i1m = (int)cm[0][ic];
      int i2m = (int)cm[1][ic];
      int i3m = (int)cm[2][ic];
      //float fls = fl[ic];//*fl[ic];

      float dx = 0.0f;

      dx += x[i3p][i2p][i1p];
      dx -= x[i3m][i2m][i1m];

      //dx *= fls;

      y[i3m][i2m][i1m] -= dx;
      y[i3p][i2p][i1p] += dx;
    }
  }

  // 3D LHS
  private static void applyLhsSlice3(
    int i3, Tensors3 d, float[][][] wp, float[][][] x, float[][][] y)
  {
    int n2 = y[0].length;
    int n1 = y[0][0].length;
    float[] di = fillfloat(1.0f,6);
    for (int i2=1; i2<n2; ++i2) {
      float[] x00 = x[i3  ][i2  ];
      float[] x01 = x[i3  ][i2-1];
      float[] x10 = x[i3-1][i2  ];
      float[] x11 = x[i3-1][i2-1];
      float[] y00 = y[i3  ][i2  ];
      float[] y01 = y[i3  ][i2-1];
      float[] y10 = y[i3-1][i2  ];
      float[] y11 = y[i3-1][i2-1];
      for (int i1=1,i1m=0; i1<n1; ++i1,++i1m) {
        if(d!=null){d.getTensor(i1,i2,i3,di);}
        float wpi = (wp!=null)?wp[i3][i2][i1]:1.0f;
        float wps = wpi*wpi;
        float d11 = di[0];
        float d12 = di[1];
        float d13 = di[2];
        float d22 = di[3];
        float d23 = di[4];
        float d33 = di[5];
        float xa = 0.0f;
        float xb = 0.0f;
        float xc = 0.0f;
        float xd = 0.0f;

        xa += x00[i1 ];
        xd -= x00[i1m];
        xb += x01[i1 ];
        xc -= x01[i1m];
        xc += x10[i1 ];
        xb -= x10[i1m];
        xd += x11[i1 ];
        xa -= x11[i1m];

        float x1 = 0.25f*(xa+xb+xc+xd)*wps;
        float x2 = 0.25f*(xa-xb+xc-xd)*wps;
        float x3 = 0.25f*(xa+xb-xc-xd)*wps;

        float y1 = d11*x1+d12*x2+d13*x3;
        float y2 = d12*x1+d22*x2+d23*x3;
        float y3 = d13*x1+d23*x2+d33*x3;

        float ya = 0.25f*(y1+y2+y3);
        float yb = 0.25f*(y1-y2+y3);
        float yc = 0.25f*(y1+y2-y3);
        float yd = 0.25f*(y1-y2-y3);

        y00[i1 ] += ya;
        y00[i1m] -= yd;
        y01[i1 ] += yb;
        y01[i1m] -= yc;
        y10[i1 ] += yc;
        y10[i1m] -= yb;
        y11[i1 ] += yd;
        y11[i1m] -= ya;  
      }
    }
  }

}
