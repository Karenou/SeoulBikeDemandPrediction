### create table
sqlcmd -S 5003.database.windows.net -d Bike -U karenou -P bigdata5003! -I -Q "
    CREATE TABLE SeoulBike
    (
        id INT NOT NULL,
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
        PRIMARY KEY (id))
    )
    ;
"

# insert data, but encounter error
bcp SeoulBike in .\SeoulBikeData.csv -S 5003.database.windows.net -d Bike -U karenou -P bigdata5003! -q -c -t ,
