import pandas as pd
import numpy as np
from pathlib import Path


def load_data(path):
	path = Path(path)
	if not path.exists():
		raise FileNotFoundError(f"CSV not found: {path}")
	return pd.read_csv(path, low_memory=False)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	# Normalize column names
	df.columns = [c.strip() for c in df.columns]

	# Replace common missing markers

	df = df.replace(['NA', 'N/A', 'Unknown', '—', '-', 'null', 'NULL'], np.nan)

	# Strip whitespace for object columns and normalize empty strings to NaN
	obj_cols = df.select_dtypes(include=['object']).columns
	for c in obj_cols:
		df[c] = df[c].astype(str).str.strip()
		df[c] = df[c].replace({'': np.nan, 'nan': np.nan})

	# Try to coerce object columns to numeric when appropriate
	for c in df.columns:
		if df[c].dtype == object:
			cleaned = df[c].astype(str).str.replace(',', '', regex=False).str.replace('%', '', regex=False)
			coerced = pd.to_numeric(cleaned, errors='coerce')
			if coerced.notna().sum() > 0:
				df[c] = coerced

	# For numeric columns, fill NaNs with the median
	num_cols = df.select_dtypes(include=[np.number]).columns
	for c in num_cols:
		median = df[c].median()
		if pd.isna(median):
			continue
		df[c] = df[c].fillna(median)

	# Drop exact duplicates and reset index
	df.drop_duplicates(inplace=True)
	df.reset_index(drop=True, inplace=True)

	return df


def save_clean(df: pd.DataFrame, out_path):
	out_path = Path(out_path)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(out_path, index=False)
	return out_path


def main():
	import argparse

	p = argparse.ArgumentParser(description='Preprocess PLACES CSV')
	p.add_argument('--input', '-i', default=r"D:\Downloads\dataset\dataset\PLACES__Local_Data_for_Better_Health,_County_Data,_2025_release_20260812.csv", help='Input CSV path')
	p.add_argument('--output', '-o', default=str(Path.cwd() / 'cleaned_places.csv'), help='Output cleaned CSV path')
	args = p.parse_args()

	csv_path = Path(args.input)
	print('Loading:', csv_path)
	df = load_data(csv_path)
	print('Original shape:', df.shape)
	clean = preprocess(df)
	print('Cleaned shape:', clean.shape)
	saved = save_clean(clean, Path(args.output))
	print('Saved cleaned CSV to', saved)
	print(clean.info())


if __name__ == '__main__':
	main()