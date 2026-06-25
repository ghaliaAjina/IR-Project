from itertools import islice

import ir_datasets
import mysql
import mysql.connector

from preprocessor import Preprocessor


class DB:
    def __init__(self):
        self.dataset = ir_datasets.load("lotte/lifestyle/dev/search")
        self.qrels = self.dataset.qrels_iter()
        self.queries = self.dataset.queries_iter()
        self.docs = self.dataset.docs_iter()
        self.db_config = {
            "host": "localhost",
            "user": "root",
            "password": "1234",
            "database": "ir_dataset",
        }

    def get_connection(self):
        return mysql.connector.connect(**self.db_config)

    def get_documents_by_ids(self, doc_ids):
        if not doc_ids:
            return {}

        placeholders = ",".join(["%s"] * len(doc_ids))
        query = f"SELECT doc_id, text FROM documents WHERE doc_id IN ({placeholders})"

        conn = self.get_connection()
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query, tuple(doc_ids))
            rows = cursor.fetchall()
            return {row["doc_id"]: row["text"] for row in rows}
        finally:
            cursor.close()
            conn.close()

    def create_db(self, cursor, sql, iterator, extract_func, batch_size=10000):
        total_inserted = 0
        while True:
            chunk = list(islice(iterator, batch_size))
            if not chunk:
                break

            data = [extract_func(item) for item in chunk]
            cursor.executemany(sql, data)
            total_inserted += len(data)        
        return total_inserted

    
    def create_tables(self): 
        try:
            db_config = {
                "host": "localhost",
                "user": "root",           
                "password": "",  
                "database": "ir_dataset"
            }    
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor()

            print("Creating tables...")
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS queries (
                    query_id VARCHAR(255) PRIMARY KEY,
                    text TEXT NOT NULL
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id VARCHAR(255) PRIMARY KEY,
                    text LONGTEXT NOT NULL
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS qrels (
                    query_id VARCHAR(255),
                    doc_id VARCHAR(255),
                    relevance INT NOT NULL,
                    iteration VARCHAR(50),
                    PRIMARY KEY (query_id, doc_id),
                    FOREIGN KEY (query_id) REFERENCES queries(query_id) ON DELETE CASCADE,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                )
            """)
            conn.commit()

            dataset_name = "lotte/lifestyle/dev/search"
            print(f"Loading dataset: {dataset_name}...")
            dataset = ir_datasets.load(dataset_name)

            
            print("Inserting queries...")
            insert_query_sql = "INSERT IGNORE INTO queries (query_id, text) VALUES (%s, %s)"
            q_count = self.insert_in_batches(
                cursor=cursor,
                sql=insert_query_sql,
                iterator=dataset.queries_iter(),
                extract_func=lambda q: (q.query_id, q.text)
            )
            print(f"✅ Processed {q_count} queries.")
            conn.commit()

            print("Inserting documents (this might take a moment depending on corpus size)...")
            insert_doc_sql = "INSERT IGNORE INTO documents (doc_id, text) VALUES (%s, %s)"
            d_count = self.insert_in_batches(
                cursor=cursor,
                sql=insert_doc_sql,
                iterator=dataset.docs_iter(),
                extract_func=lambda d: (d.doc_id, d.text)
            )
            print(f"✅ Processed {d_count} documents.")
            conn.commit()

            print("Inserting qrels...")
            insert_qrel_sql = "INSERT IGNORE INTO qrels (query_id, doc_id, relevance, iteration) VALUES (%s, %s, %s, %s)"
            qr_count = self.insert_in_batches(
                cursor=cursor,
                sql=insert_qrel_sql,
                iterator=dataset.qrels_iter(),
                extract_func=lambda qr: (qr.query_id, qr.doc_id, qr.relevance, qr.iteration)
            )
            print(f"✅ Processed {qr_count} qrels.")
            conn.commit()

            print("All data successfully stored in MySQL!")

        except mysql.connector.Error as err:
            print(f"MySQL Error: {err}")
            if 'conn' in locals() and conn.is_connected():
                conn.rollback()

        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals() and conn.is_connected():
                conn.close()
                print("MySQL connection closed.")


        def preprocess_db():
            try:
                conn = mysql.connector.connect(**db_config)
                cursor_read = conn.cursor(dictionary=True)
                cursor_write = conn.cursor()

                preprocessor = Preprocessor()
                print("Ensuring preprocessed tables exist...")
                
                cursor_write.execute("""
                    CREATE TABLE IF NOT EXISTS preprocessed_queries ( 
                        query_id VARCHAR(255) PRIMARY KEY,
                        text_clean TEXT NOT NULL,
                        FOREIGN KEY (query_id) REFERENCES queries(query_id) ON DELETE CASCADE
                    )
                """)

                cursor_write.execute("""
                    CREATE TABLE IF NOT EXISTS preprocessed_documents (
                        doc_id VARCHAR(255) PRIMARY KEY,
                        text_clean LONGTEXT NOT NULL,
                        FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                    )
                """)
                conn.commit()

                def process_table(source_table, target_table, id_col, text_col, batch_size=5000):
                    print(f"\nProcessing {source_table} -> {target_table}...")
                    
                    cursor_read.execute(f"SELECT COUNT(*) as total FROM {source_table}")
                    total_rows = cursor_read.fetchone()['total']
                    print(f"Total rows to process: {total_rows}")

                    offset = 0
                    total_inserted = 0
                    
                    while offset < total_rows:
                        cursor_read.execute(f"SELECT {id_col}, {text_col} FROM {source_table} LIMIT %s OFFSET %s", (batch_size, offset))
                        rows = cursor_read.fetchall()
                        
                        if not rows:
                            break
                        
                        processed_data = []
                        for row in rows:
                            record_id = row[id_col]
                            raw_text = row[text_col]
                            
                            if raw_text:
                                # Use your class to get tokens and join them back into a string
                                tokens = preprocessor.process(raw_text)
                                clean_text = " ".join(tokens)
                            else:
                                clean_text = ""
                                
                            processed_data.append((record_id, clean_text))
                        
                        insert_sql = f"INSERT IGNORE INTO {target_table} ({id_col}, text_clean) VALUES (%s, %s)"
                        cursor_write.executemany(insert_sql, processed_data)
                        conn.commit()
                        
                        total_inserted += len(processed_data)
                        offset += batch_size
                        print(f"  ... Processed {total_inserted} / {total_rows}")

                
                process_table(
                    source_table="queries",
                    target_table="preprocessed_queries",
                    id_col="query_id",
                    text_col="text"
                )

                process_table(
                    source_table="documents",
                    target_table="preprocessed_documents",
                    id_col="doc_id",
                    text_col="text"
                )

                print("\nAll preprocessing tasks completed successfully!")

            except mysql.connector.Error as err:
                print(f"MySQL Error: {err}")
                if 'conn' in locals() and conn.is_connected():
                    conn.rollback()

            finally:
                if 'cursor_read' in locals(): cursor_read.close()
                if 'cursor_write' in locals(): cursor_write.close()
                if 'conn' in locals() and conn.is_connected():
                    conn.close()
                    print("MySQL connection closed.")