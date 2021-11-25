# SeoulBikeDemandPrediction
MSBD5003 course project

1. Set up connection to virtual machine on Microsoft Azure
```
# Generate ssh private and public key
ssh-keygen -t rsa -b 4096

# Create Virtual Machine on Microsoft Azure
# vm1 ip: 52.149.147.

# Connect to VM1 (root)
ssh -i ~/.ssh/id_rsa azureuser@52.149.147.35
```


## Following steps are done in virtual machine
2. Install hadoop, spark, anaconda; configure jupyter-notebook
```
# install spark
wget https://downloads.apache.org/spark/spark-3.0.3/spark-3.0.3-bin-hadoop2.7.tgz
tar xf spark-3.0.3-bin-hadoop2.7.tgz

# install java
sudo add-apt-repository ppa:openjdk-r/ppa
sudo apt-get update
sudo apt-get install openjdk-8-jdk

# install python
sudo apt-get install python3

# install hadoop and set up distributed nodes
wget https://dlcdn.apache.org/hadoop/common/hadoop-2.10.1/hadoop-2.10.1.tar.gz
tar xf hadoop-2.10.1.tar.gz

# add the following cmd in /hadoop/etc/hadoop/hadoop-env.sh
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/jre
export HADOOP_CONF_DIR=/home/azureuser/hadoop-2.10.1/etc/hadoop

# add the following cmd in /hadoop/etc/hadoop/core-site.xml
<configuration>
        <property>
            <name>fs.default.name</name>
            <value>hdfs://vm1:9000</value>
        </property>
</configuration>

# add the following cmd in /hadoop/etc/hadoop/hdfs-site.xml
<configuration>
        <property>
                <name>dfs.namenode.name.dir</name>
                <value>/home/azureuser/data/nameNode</value>
        </property>

        <property>
                <name>dfs.datanode.data.dir</name>
                <value>/home/azureuser/data/dataNode</value>
        </property>

        <property>
                <name>dfs.replication</name>
                <value>3</value>
        </property>

        <property>
                <name>dfs.block.size</name>
                <value>16777216</value>
        </property>
                
        <property>
                <name>dfs.namenode.datanode.registration.ip-hostname-check</name>             
                <value>false</value>
        </property>
</configuration>

# start namenode
./sbin/hadoop-daemon.sh start namenode


# add in /hadoop/etc/hadoop/slaves
vm1
vm2
vm3

```

3. Configure Jupyter-notebook so that we could use the interface on own ocmputer
```
# open notebook on own computer via another terminal, replace the user name and ip of the vm
ssh -N -f -L localhost:8890:localhost:8890 azureuser@52.149.147.35
```

4. Install mysql, download mysql-connector and move to $SPARK_HOME/jars
```
# install mysql-server
sudo apt install mysql-server

# set password for root user
sudo mysql_secure_installation
sudo systemctl restart mysql

# move the csv file under this folder so that it can be imported in the mysql database
sudo chmod 777 /var/lib/mysql-files
```

5. Load data into mysql
```
Create database big_data_5003;

create table SeoulBike ( 
id INT NOT NULL AUTO_INCREMENT, 
Date VARCHAR(50) NOT NULL, 
RentedBikeCount SMALLINT NOT NULL, 
Hour TINYINT NOT NULL, 
Temperature FLOAT, 
Humidity_pct TINYINT, 
WindSpeed_m_per_s FLOAT, 
Visibility_10m SMALLINT, 
DewPointTemperature FLOAT, 
SolarRadiation FLOAT, 
Rainfall_mm FLOAT, 
Snowfall_cm FLOAT, 
Seasons VARCHAR(50), 
Holiday VARCHAR(50), 
FunctioningDay VARCHAR(50),
PRIMARY KEY (id) 
);

LOAD DATA INFILE '/var/lib/mysql-files/SeoulBikeData.csv' 
INTO TABLE SeoulBike 
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"' 
LINES TERMINATED BY '\r\n'
IGNORE 1 ROWS;
```

6. Run the main.py, start reading and training the ml models

