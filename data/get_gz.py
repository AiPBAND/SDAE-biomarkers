import gzip

def file_name(fn):
    with gzip.open(fn, 'rb') as f:
        file_content = f.read()
        return file_content