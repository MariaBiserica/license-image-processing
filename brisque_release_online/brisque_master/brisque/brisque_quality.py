import cv2
from repo.brisque_release_online.brisque_master.brisque.brisque_algorithm import BRISQUE


# Funcția care testează imaginea
def measure_brisque():
    # Definește calea către imagine
    image_path = "..\\..\\..\\VGG16\\data\\512x384\\826373.jpg"

    # Încarcă imaginea folosind OpenCV
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Verifică dacă imaginea a fost încărcată corect
    assert img is not None, "Eroare la încărcarea imaginii. Verifică calea."

    # Creează un obiect BRISQUE
    obj = BRISQUE(url=False)

    # Calculează scorul BRISQUE
    score = obj.score(img)
    print(f"\nScorul BRISQUE al imaginii este: {score}")


# Dacă vrei să păstrezi și funcția main
def main():
    measure_brisque()


if __name__ == "__main__":
    main()
