# central limit theorem visualizer

demonstrates how sample means become normal regardless of the original distribution shape.

## quick start

```bash
git clone https://github.com/pratyakshpatel/clt-visualiser.git
cd clt-visualiser
python main.py
```

no dependencies needed for the basic demo!

## what it shows

the central limit theorem says: take any distribution (even weird shaped ones), sample from it many times, and the sample means will always form a bell curve.

this project proves it with:
- exponential distribution (very skewed) → normal sample means
- visual histogram showing the bell curve
- comparison of theory vs actual results
- effect of different sample sizes

## running options

**option 1: basic demo (recommended)**
```bash
python main.py
```
uses only python standard library, shows text-based histogram

**option 2: simple version**
```bash
python execute_demo.py
```
same as main.py but slightly different format

**option 3: full interactive version**
```bash
pip install -r requirements.txt
python clt_visualizer.py
```
has sliders and live plots (needs jupyter widgets)

**option 4: jupyter notebook**
```bash
pip install jupyter numpy matplotlib scipy ipywidgets
jupyter lab CLT_Visualizer.ipynb
```
best experience with interactive plots

## what you'll see

```
central limit theorem demonstration
generating 1000 samples of size 30...
results:
original distribution: exponential (right-skewed)
population mean: 1.0000
expected std error: 0.1826
observed from 1000 samples:
mean of sample means: 0.9987
std error: 0.1834
excellent match!

histogram of sample means:
 0.47- 0.54 |██ (2)
 0.54- 0.61 |████ (5)
 0.61- 0.68 |████████ (12)
 ...bell curve shape...
```

## files

- `main.py` - clean demo, run this first
- `execute_demo.py` - alternative demo version  
- `clt_visualizer.py` - full interactive version with plots
- `CLT_Visualizer.ipynb` - jupyter notebook
- `requirements.txt` - dependencies for advanced features

## the math

- original distribution: any shape
- sample size: n  
- number of samples: many
- result: sample means ~ Normal(μ, σ/√n)

works for any distribution - that's the magic of CLT!