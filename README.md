# GRADIATOR
The simple CLI color gradient applicator.

## Requirements:
  - Python3
  - OpenCV2

The program *should* run on Windows, MacOS and Linux, but I've only tested it on linux.

## Usage:

python3 gradiator.py [options] src dst

### Options are:

    -h --help (self-explanatory)
  Colors:
    
    -c1 [valid six digit hex number]
    
    -c2 [valid six digit hex number]
    
  Angle:
  
    -a [angle at which the gradient should be applied in degrees]
    
  Display:

    -s --show (display new image after program execution)
    
  Mode:
  
    -n Normal (default)
    
    -d Dark (apply gradient on darker parts of image)
    
    -i Invert (apply gradient on inverted image)
    
    -w Weird (yup)

  
