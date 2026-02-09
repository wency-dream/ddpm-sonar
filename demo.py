from pathlib import Path;
exts = ['.jpg', '.jpeg', '.png', '.tiff','.bmp']
paths =[str(f.absolute()) for f in Path('E:\\data_classfree').rglob('*') if f.suffix.lower() in exts]
print(paths)