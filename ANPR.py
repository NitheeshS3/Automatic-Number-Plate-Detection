import os
import glob
import numpy as np
import cv2
from PIL import Image
import pytesseract
import re

# Number Plate Detection Function
def number_plate_detection(img):
    def preprocess_plate(plate_img):
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return thresh

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(blurred, 30, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = w / float(h)
        area = cv2.contourArea(cnt)

        if 2 < ratio < 6 and 1000 < area < 100000:
            plate_img = img[y:y + h, x:x + w]
            plate_crop = preprocess_plate(plate_img)
            plate_pil = Image.fromarray(plate_crop)
            text = pytesseract.image_to_string(
                plate_pil,
                config='--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            ).strip()

            # Clean and validate
            cleaned = "".join(re.split("[^A-Z0-9]", text.upper()))
            if len(cleaned) >= 5:
                return cleaned
    return None

# Quick sort
def quickSort(arr, low, high):
    def partition(arr, low, high):
        i = low - 1
        pivot = arr[high]
        for j in range(low, high):
            if arr[j] < pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1

    if low < high:
        pi = partition(arr, low, high)
        quickSort(arr, low, pi - 1)
        quickSort(arr, pi + 1, high)
    return arr

# Binary search
def binarySearch(arr, l, r, x):
    if r >= l:
        mid = l + (r - l) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] > x:
            return binarySearch(arr, l, mid - 1, x)
        else:
            return binarySearch(arr, mid + 1, r, x)
    return -1

# Main program
def main():
    print("HELLO!!")
    print("Welcome to the Number Plate Detection System.\n")

    array = []
    dataset_path = os.path.expanduser("~/Desktop/sem6 projects/ANPR/Dataset/*.jpeg")

    for img_path in glob.glob(dataset_path):
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_resized = cv2.resize(img, (1080, 720))
        cv2.imshow("Image of car", img_resized)
        cv2.waitKey(250)
        cv2.destroyAllWindows()

        number_plate = number_plate_detection(img)
        if number_plate:
            array.append(number_plate)
            print(f"{number_plate}")
        else:
            print(f"❌ Could not detect in: {os.path.basename(img_path)}")

    if array:
        array = quickSort(array, 0, len(array) - 1)

    print("\nThe Vehicle numbers registered are:")
    for num in array:
        print(num)

    print("\n")

    # image search
    search_img_path = os.path.expanduser("~/Desktop/sem6 projects/ANPR/Search_Image/2.jpeg")
    img = cv2.imread(search_img_path)
    if img is not None:
        img_resized = cv2.resize(img, (1080, 720))
        cv2.imshow("Search Image", img_resized)
        cv2.waitKey(500)
        cv2.destroyAllWindows()

        number_plate = number_plate_detection(img)
        if number_plate:
            print("🔍 The car number found in search image is:", number_plate)
            result = binarySearch(array, 0, len(array) - 1, number_plate)
            if result != -1:
                print("✅ The Vehicle is ALLOWED to visit.\n")
            else:
                print("❌ The Vehicle is NOT allowed to visit.\n")
        else:
            print("❌ No valid number plate found in the search image.\n")
    else:
        print("❌ Search image not found.\n")

if __name__ == "__main__":
    main()