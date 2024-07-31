from pathlib import Path
import fitz

dir_path = Path(r'D:\Data\chicago\jiongjiong')


input_items = [
    {
        'input_file_name': r'master_sjtu.pdf',
        'pages': 4, # First numbers of pages.
        'output_file_name': 'JiongjiongLi_master_sjtu_grad_cert.pdf'
    },
    {
        'input_file_name': r'JiongjiongLi_certs_transcripts_master_sjtu.pdf',
        'pages': 8, # First numbers of pages.
        'output_file_name': 'JiongjiongLi_creds_master_sjtu.pdf'
    },
    {
        'input_file_name': r'JiongjiongLi_certs_transcripts_bachelor_ecnu.pdf',
        'pages': 8, # First numbers of pages.
        'output_file_name': 'JiongjiongLi_creds_bachelor_ecnu.pdf'
    },
]

for input_item in input_items:
    input_file_path = dir_path / input_item['input_file_name']
    output_file_path = dir_path / input_item['output_file_name']
    pages = input_item['pages']

    with fitz.open(input_file_path.as_posix()) as input_file:
        input_file.select(range(pages))
        input_file.save(output_file_path.as_posix())


