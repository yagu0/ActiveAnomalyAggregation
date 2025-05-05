# Active Anomaly Aggregation

## Requirements

Python 3 (at least 3.8, up to 3.12 included).
Python's package manager pip is required for the installation methods described below.

## Installation

Here are a few different ways to download and install the library from GitHub.  

### Method 1

In theory the easiest way to download the library is to open a terminal window on your computer and type:

    pip install git+https://github.com/yagu0/ActiveAnomalyAggregation.git

If you want the library to be accessible to one specific Python environment you have created locally,
the basic rule is that you need to activate that environment before running the pip command above. On
MacOS/Linux, this probably means typing something like:

    source /path_to_your_environment/bin/activate

while on Windows try:

    \path_to_your_environment\Scripts\activate

If instead you are using the Anaconda Navigator App GUI with locally created environments (or just the base root environment),
click on 'Environments' and then click on the environment you wish to make active. Then click on the arrow that
appears next to the environment's name and select 'Open terminal'. You can then paste, in the terminal window that opens, 
the pip install command above.

### Method 2

Click on the green 'Code' button on this GitHub page: 

    https://github.com/yagu0/ActiveAnomalyAggregation/tree/main 
    
and download the .zip file. 

Unzip the file on your computer. You then need to open a terminal window (if you are using the Anaconda Navigator App GUI, first
open a terminal window inside your active environment as described above). 

If necessary, type 'ls' into the terminal window to see what folder you are inside currently. You then need to tell the terminal
window to move from where you currently are to inside the main outer folder of the unzipped library on your computer. If you don't
know how to do this, here are some of the basics for using a terminal on macOS/Linux:

    https://terminalcheatsheet.com/guides/navigate-terminal

Now that you are inside the main outer folder of the library, you can simply type into the terminal window:

    pip install .

Yes, the '.' is part of the command. 

### Method 3

A third way to install the library is to first clone (i.e., copy) the library from GitHub by running the following line in a terminal window

    git clone https://github.com/yagu0/ActiveAnomalyAggregation.git

See above for how to do this for only a specific environment. Since the library is now downloaded to the current directory you
are in on your computer, it suffices to move from the current directory inside the downloaded library with the command:

    cd ActiveAnomalyAggregation

Finally, as above, enter:

    pip install .

Yes, the '.' is part of the command. 

## Usage

See code in doc/source/content/examples.
