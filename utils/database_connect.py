
import mysql.connector
import configparser

# Read the configuration file
config = configparser.ConfigParser()
config.read('config/config.ini')

#########################################
#### Connecting with MySQL Database #####
#########################################



conn = mysql.connector.connect(
        host=config['database']['host'],
        user=config['database']['user'],
        password=config['database']['password']
    )
cursor = conn.cursor()
cursor.execute("CREATE DATABASE IF NOT EXISTS testing")
cursor.close()

cursor = conn.cursor()
cursor.execute("USE testing")
cursor.close()

cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS newuser (
    name VARCHAR(50) NOT NULL,
    embedding TEXT NOT NULL
);
                )''')
cursor.close()
    

def initialize_db():
    conn = mysql.connector.connect(
        host=config['database']['host'],
        user=config['database']['user'],
        password=config['database']['password'],
                database=config['database']['database']
            )
    return conn
#########################################
#### Defining the Required Functions #####

def check_user(name, conn):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM newuser WHERE name=%s", (name,))
    if cursor.fetchone():
        return 1
    else:
        return 0
    
########################################

def update_username(original_username, new_username, conn):
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM newuser WHERE name=%s", (original_username,))
            if cursor.fetchone():
                cursor.execute("UPDATE newuser  SET name = %s WHERE name = %s", (new_username, original_username))
                conn.commit()  
                return 1
            else:
                return 0
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return 0
    finally:
        conn.close() 

########################################

def find_delete(name, conn):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM newuser WHERE name=%s", (name,))
    user_info = cursor.fetchone()
    if user_info:
        cursor.execute("DELETE FROM newuser WHERE name=%s", (name,))
        conn.commit()
        content = 'User ' + str(name) + ' deleted successfully'
        return content,1
    else:
        content = 'User ' + str(name) + ' not present'
        return content,0

########################################

def input_data(image_name, embeddings, conn):
    cursor = conn.cursor()
    cursor.execute("INSERT INTO newuser (name, embedding) VALUES (%s, %s)", (image_name, embeddings))
    conn.commit()
    return 0

########################################

def get_users_from_database(conn):
    try:
        cursor = conn.cursor()
        cursor = conn.cursor(dictionary=True)  # Return results as dictionaries
        cursor.execute("SELECT name FROM newuser")
        users = cursor.fetchall()
        conn.close()
        return users
    except Exception as e:
        return str(e)
