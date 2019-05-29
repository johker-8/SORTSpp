#!/usr/bin/env bash

sudo dkpg --configure -a
sudo apt-get install git
sudo apt-get install python-dev

python_v="`python -c 'import sys; print(sys.version_info[0])'`"

if [ "$python_v" -ne "2" ]; then
  python_v="`python2.7 -c 'import sys; print(sys.version_info[0])'`"

  if [ "$python_v" -ne "2" ]; then
    exit 1
  fi
fi

sudo apt-get install python-pip
sudo apt-get install python-tk
sudo apt-get install libfreetype6-dev
sudo apt-get install libpng12-dev
sudo apt-get install libopenmpi-dev
pip install virtualenv
virtualenv env2.7
source env2.7/bin/activate
pip install --upgrade pip
pip install -r pip_req.txt

#orekit
mkdir orekit_build
cd orekit_build/

sudo apt-get install openjdk-8-jdk
sudo apt install maven

pip install jcc
pip install scipy
pip install matplotlib
pip install pytest

export JCC_JDK="/usr/lib/jvm/java-8-openjdk-amd64"
export SRC_DIR="`pwd`"

git clone https://github.com/petrushy/Orekit.git
cd Orekit/
mvn package
cd ..
mv Orekit/target/*.jar .

git clone https://gitlab.orekit.org/orekit-labs/python-wrapper.git
mv python-wrapper/python_files/* .


curl https://www.hipparchus.org/downloads/hipparchus-1.3-bin.zip
tar -xvf hipparchus-1.3-bin.zip
mv hipparchus-1.3-bin/*.jar .

python -m jcc \
--use_full_names \
--python orekit \
--version 9.3 \
--jar $SRC_DIR/orekit-9.2.jar \
--jar $SRC_DIR/hipparchus-core-1.3.jar \
--jar $SRC_DIR/hipparchus-filtering-1.3.jar \
--jar $SRC_DIR/hipparchus-fitting-1.3.jar \
--jar $SRC_DIR/hipparchus-geometry-1.3.jar \
--jar $SRC_DIR/hipparchus-ode-1.3.jar \
--jar $SRC_DIR/hipparchus-optim-1.3.jar \
--jar $SRC_DIR/hipparchus-stat-1.3.jar \
--package java.io \
--package java.util \
--package java.text \
--package org.orekit \
java.io.BufferedReader \
java.io.FileInputStream \
java.io.FileOutputStream \
java.io.InputStream \
java.io.InputStreamReader \
java.io.ObjectInputStream \
java.io.ObjectOutputStream \
java.io.PrintStream \
java.io.StringReader \
java.io.StringWriter \
java.lang.System \
java.text.DecimalFormat \
java.text.DecimalFormatSymbols \
java.util.ArrayList \
java.util.Arrays \
java.util.Collection \
java.util.Collections \
java.util.Date \
java.util.HashMap \
java.util.HashSet \
java.util.List \
java.util.Locale \
java.util.Map \
java.util.Set \
java.util.TreeSet \
--module $SRC_DIR/pyhelpers.py \
--reserved INFINITE \
--reserved ERROR \
--reserved OVERFLOW \
--reserved NO_DATA \
--reserved NAN \
--reserved min \
--reserved max \
--reserved mean \
--build \
--install


cd test/
python -m pytest

cd ..
cd ..



#basemap
mkdir basemap_build
cd basemap_build

sudo apt-get install proj-bin

wget --no-check-certificate https://github.com/matplotlib/basemap/archive/master.tar.gz
tar -xvf master.tar.gz
export GEOS_DIR=

./configure --prefix=$GEOS_DIR
make; make install

python setup.py install