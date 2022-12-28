import os
file_path = r"C:\Users\freet\Desktop\3학년2학기\인공지능\과제\꽃 분류\train_data_rotate\marigold"
file_name = os.listdir(file_path)

i = 1  # 파일에 들어가는 숫자
cnt = 1  # 돌아가는 횟수
for name in file_name:
    src = os.path.join(file_path, name)
    path, ext = os.path.splitext(name)

    dst = 'marigold' + str(i) + ext
    print(dst)
    dst = os.path.join(file_path, dst)
    os.rename(src, dst)
    cnt += 1
    i += 1
