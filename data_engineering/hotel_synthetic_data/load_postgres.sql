-- ============================================================
-- Synthetic Hotel Reviews Dataset (Postgres-friendly)
-- ============================================================
-- Tables:
--   hotels(hotel_id, hotel_name, city, country, price_tier)
--   hotel_labels(hotel_id, label)
--   reviews(review_id, hotel_id, user_id, rating, review_date, is_verified)
-- ============================================================

DROP TABLE IF EXISTS reviews;
DROP TABLE IF EXISTS hotel_labels;
DROP TABLE IF EXISTS hotels;

CREATE TABLE hotels (
  hotel_id    INT PRIMARY KEY,
  hotel_name  TEXT NOT NULL,
  city        TEXT NOT NULL,
  country     TEXT NOT NULL,
  price_tier  TEXT NOT NULL
);

CREATE TABLE hotel_labels (
  hotel_id INT NOT NULL,
  label    TEXT NOT NULL,
  PRIMARY KEY (hotel_id, label),
  FOREIGN KEY (hotel_id) REFERENCES hotels(hotel_id)
);

CREATE TABLE reviews (
  review_id   INT PRIMARY KEY,
  hotel_id    INT NOT NULL,
  user_id     INT NOT NULL,
  rating      INT NOT NULL CHECK (rating BETWEEN 0 AND 5),
  review_date DATE NOT NULL,
  is_verified INT NOT NULL CHECK (is_verified IN (0,1)),
  FOREIGN KEY (hotel_id) REFERENCES hotels(hotel_id)
);

-- Load CSVs (edit paths to your local machine):
COPY hotels       FROM './data_engineering/hotel_synthetic_data/hotels.csv'       WITH (FORMAT csv, HEADER true);
COPY hotel_labels FROM './data_engineering/hotel_synthetic_data/hotel_labels.csv' WITH (FORMAT csv, HEADER true);
COPY reviews      FROM './data_engineering/hotel_synthetic_data/reviews.csv'      WITH (FORMAT csv, HEADER true);
