# Star Field Image Simulator

![Tests](https://github.com/peeeyow/star-field-image-simulator/actions/workflows/tests.yaml/badge.svg)

Simulator that generates star field images based on a given set of configuration. 

## Parameters

- Star magnitude limit (M<sub>V</sub> ≥ -1.0876)
- Quantity of missing stars (n<sub>missing</sub> ≥ 0)
- Unexpected False Stars
  - Quantity of false stars (n<sub>false</sub> ≥ 0)
  - Star magnitude limit of false stars (-1.0876 ≤ M<sub>V,false</sub> ≤ 6.00)
- Size of FOV
  - Horizontal FOV (5° ≤ FOV<sub>x</sub> ≤ 25°)
  - Vertical FOV(5° ≤ FOV<sub>y</sub> ≤ 25°) 
- Sensor orientation
  - Right ascension (0° ≤ α<sub>0</sub> ≤ 360°)
  - Declination (-90° ≤ δ<sub>0</sub> ≤ 90°)
  - Roll (-90° ≤ φ<sub>0</sub> ≤ 90°)
- Image resolution
  - Horizontal resolution (256 ≤ RES<sub>x</sub> ≤ 2048)
  - Vertical resolution (256 ≤ RES<sub>y</sub> ≤ 2048)
- Point Spread Function constants
  - Defocus Level (σ > 0)
  - Star intensity constant (C > 0)
- Position noise constant (λ ≥ 0)
- Random noise
  - Dark current noise mean intensity constant (0 ≤ η<sub>DC</sub> ≤ 1)
  - Dark current noise time constant (τ<sub>DC</sub> ≥ 0)
  - Read noise mean intensity value (0 ≤ η<sub>RN</sub> ≤ 1)
  - Shot noise standard deviation ( σ<sub>SN</sub> ≥ 0)
- Discrete Canvass Computation Function (integrated := boolean)
- Lazy mode (lazy := boolean)

## Sample Images
Attitude: (20°, 20°, 90°)
### Clean Image
![clean](https://user-images.githubusercontent.com/69317890/161408424-4e4ca837-e3ab-422f-95c6-0f1ceaceeb1b.png)
### With Dark Current Noise
![image](https://user-images.githubusercontent.com/69317890/161408495-8c288178-2f00-497e-821b-31beea61a9d7.png)
### With Read Noise
![image](https://user-images.githubusercontent.com/69317890/161408511-2639d2e5-a42b-441e-8eac-cef78fb97156.png)
### With Shot Noise
![image](https://user-images.githubusercontent.com/69317890/161408518-fa2cdfac-9865-4d9b-aa9b-72adab36eb3d.png)
### With Position Noise
![image](https://user-images.githubusercontent.com/69317890/161408506-11d69ce1-93a5-4e75-ab39-461ce2c2308f.png)

