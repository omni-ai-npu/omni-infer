export PYTHONWARNING=ignore
xgramr=$(pip show xgrammar | grep Location: | cut -d ' ' -f 2-)
modif_file=$(find "$xgramr" -name apply_token_bitmask_inplace_torch_compile.py | awk 'NR==1')
sed -i 's/@torch.compile(dynamic=True)//g' "$modif_file"
