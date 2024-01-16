1- Create a virtual environment by running the following command:

python3 -m venv 'your env name'


2- To activate your virtual environment, run the code below:

source 'your env name'/bin/activate


3- Install dependencies:

sudo apt install python3-opencv

pip install --upgrade pip

pip install -r requirements.txt


4- Run the service using one of the following commands:

python api.py

waitress-serve --port=5000 --call api:create_app
