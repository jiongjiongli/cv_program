from pathlib import Path
import fitz


dir_path = Path(r'D:\Data\chicago\jiongjiong')


input_pdf_dict = {
    # 'JiongjiongLi_certs_transcripts_bachelor_ecnu': [
    #     r'bachelor_ecnu.pdf',
    #     r'scholaro-gpa-bachelor-ecnu.pdf',
    # ],
    # 'JiongjiongLi_certs_transcripts_master_sjtu': [
    #     {
    #         'file_name': r'master_sjtu.pdf',
    #         'pages': 4, # First numbers of pages.
    #     },
    #     r'JiongjiongLi_gpa_sjtu_cn_print.pdf',
    #     r'JiongjiongLi_gpa_sjtu_en_print.pdf',
    #     r'scholaro-gpa-master-sjtu.pdf',
    # ],
    'JiongjiongLi_creds_bachelor_shu': [
        {
            'file_name': r'freshman_admission_list_cn.pdf',
            'rotate': 270, # Rotate degree.
        },
        r'freshman_admission_list_en_stamp.pdf',
        r'su-transcripts.pdf',
        r'JiongjiongLi_transcript_shu_en_stamp.pdf',
    ],
}



for key, file_paths in input_pdf_dict.items():
    output_pdf_path = dir_path / r'{}.pdf'.format(key)

    output_pdf = fitz.open()

    for item in file_paths:
        pages = None
        rotate = None

        if isinstance(item, str):
            input_file_name = item
        else:
            input_file_name = item['file_name']
            if 'pages' in item:
                pages = range(item['pages'])

            if 'rotate' in item:
                rotate = item['rotate']

        input_file_path = dir_path / input_file_name

        with fitz.open(input_file_path.as_posix()) as input_file:
            if pages:
                input_file.select(pages)

            if rotate:
                for page in input_file:
                    page.set_rotation(page.rotation + rotate)

            output_pdf.insert_pdf(input_file)

    output_pdf.save(output_pdf_path.as_posix())
    print(r'Generated: {}'.format(output_pdf_path))
