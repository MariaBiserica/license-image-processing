import cv2
from repo.brisque_release_online.brisque_master.brisque.brisque_algorithm import BRISQUE


# Funcția care testează imaginea
def measure_brisque(img_path):
    # Încarcă imaginea folosind OpenCV
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # Verifică dacă imaginea a fost încărcată corect
    assert img is not None, "Eroare la încărcarea imaginii. Verifică calea."

    # Creează un obiect BRISQUE
    obj = BRISQUE(url=False)

    # Calculează scorul BRISQUE
    quality_score = obj.score(img)

    return quality_score


# Dacă vrei să păstrezi și funcția main
def main():
    image_path = "..\\..\\..\\VGG16\\data\\512x384\\826373.jpg"
    quality_score = measure_brisque(image_path)
    print(f'BRISQUE Quality Score: {quality_score:.4f}')


if __name__ == "__main__":
    main()
