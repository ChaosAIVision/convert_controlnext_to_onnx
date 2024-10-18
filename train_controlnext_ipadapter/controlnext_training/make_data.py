import csv
import os

def create_csv_from_folders(image_folders, output_csv):
    """
    Tạo file CSV từ tất cả các tấm ảnh trong 5 thư mục, mỗi thư mục tương ứng với một cột.
    
    Parameters:
    - image_folders (list): Danh sách chứa đường dẫn đến 5 thư mục chứa ảnh.
    - output_csv (str): Đường dẫn đến file CSV đầu ra.
    """
    if len(image_folders) != 5:
        raise ValueError("Cần phải có đúng 5 thư mục trong danh sách image_folders.")
    
    image_paths = []

    # Lặp qua từng thư mục để lấy ảnh
    for folder in image_folders:
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Thư mục {folder} không tồn tại.")
        
        # Lấy danh sách các tệp trong thư mục
        folder_images = []
        for file_name in os.listdir(folder):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_file = os.path.join(folder, file_name)
                folder_images.append(image_file)

        # Nếu không tìm thấy ảnh, thêm None vào danh sách
        if not folder_images:
            folder_images.append(None)  # Thêm None nếu không có ảnh

        image_paths.append(folder_images)

    # Tạo file CSV và ghi dữ liệu
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Tạo tiêu đề cho các cột
        header = ["pixel_values", "conditioning_pixel_values", "original_values", "mask_values", "ipadapter_images", "text_prompt"]
        writer.writerow(header)

        # Ghi nội dung vào hàng (tất cả đường dẫn ảnh + text prompt)
        text_prompt = "High quality, realistic"
        
        # Chuyển đổi từng danh sách ảnh thành hàng trong CSV
        max_length = max(len(paths) for paths in image_paths)
        for i in range(max_length):
            row = []
            for paths in image_paths:
                if i < len(paths):
                    row.append(paths[i])
                else:
                    row.append(None)  # Thêm None nếu không còn ảnh
            row.append(text_prompt)  # Thêm text prompt
            writer.writerow(row)
    
    print(f"File CSV đã được tạo thành công: {output_csv}")

# Ví dụ sử dụng hàm
image_folders = [
    "/home/chaos/Desktop/target_image/",
    "/home/chaos/Desktop/controlnext_cond/",
    "/home/chaos/Desktop/original_mask/",
    "/home/chaos/Desktop/masked_image/",
    "/home/chaos/Desktop/ip_adapter_image/"
]
output_csv = "/home/chaos/Desktop/data/data.csv"

create_csv_from_folders(image_folders, output_csv)
