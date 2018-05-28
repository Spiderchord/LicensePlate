// DetectChars.cpp

#include "DetectChars.h"


cv::Ptr<cv::ml::KNearest> kNearest = cv::ml::KNearest::create();


bool loadKNNDataAndTrainKNN(void) {

    //odczytywanie classification numbers

    cv::Mat matClassificationInts;              //wczytujemy classification numbers do macierzy

    cv::FileStorage fsClassifications("classifications.xml", cv::FileStorage::READ);        //otwieramy plik classifications.xml

    if (fsClassifications.isOpened() == false) {                                                        //wyjdz z programu jesli nie mozna otworzyc pliku
        std::cout << "Nie mozna otworzyc pliku classifications.xml !\n\n";        
        return(false);                                                                                  
    }

    fsClassifications["classifications"] >> matClassificationInts;          //wczytywanie danych z pliku do macierzy
    fsClassifications.release();                                            //zamykanie pliku

                                                                            //wczytywanie training images

    cv::Mat matTrainingImagesAsFlattenedFloats;         //tu wczytujemy obrazki z pliku images.xml

    cv::FileStorage fsTrainingImages("images.xml", cv::FileStorage::READ);              //otwieranie pliku

    if (fsTrainingImages.isOpened() == false) {                                                 //wyjdz z programu jesli nie mozna otworzyc pliku
        std::cout << "Nie mozna otworzyc pliku images.xml !\n\n";         
        return(false);                                                                          
    }

    fsTrainingImages["images"] >> matTrainingImagesAsFlattenedFloats;           //wczytywanie obrazkow do macierzy
    fsTrainingImages.release();                                                 //zamknij plik

    //UCZENIE                                                                                        
                                                                                
    kNearest->setDefaultK(1);

    kNearest->train(matTrainingImagesAsFlattenedFloats, cv::ml::ROW_SAMPLE, matClassificationInts);
	
    return true;
}

std::vector<PossiblePlate> detectCharsInPlates(std::vector<PossiblePlate> &vectorOfPossiblePlates) {
    int intPlateCounter = 0;				//do pokazywania kolejnych krokow
    cv::Mat imgContours;
    std::vector<std::vector<cv::Point> > contours;
    cv::RNG rng;

    if (vectorOfPossiblePlates.empty()) {               //wyjdz z funkcji jak nie ma mozliwych tablic
        return(vectorOfPossiblePlates);                 
    }
    //tu ju¿ wiemy, ¿e vectorOfPossiblePlates ma jak¹œ tablice

    for (auto &possiblePlate : vectorOfPossiblePlates) {            //dla kazdej mozliwej tablicy

        preprocess(possiblePlate.imgPlate, possiblePlate.imgGrayscale, possiblePlate.imgThresh);        //preprocess

        //skalowanie do 60% w celu lepszego rozpoznawania
        cv::resize(possiblePlate.imgThresh, possiblePlate.imgThresh, cv::Size(), 1.6, 1.6);

        // threshold jeszcze raz aby usunac szare obszary
        cv::threshold(possiblePlate.imgThresh, possiblePlate.imgThresh, 0.0, 255.0, CV_THRESH_BINARY | CV_THRESH_OTSU);
        
		//znajdujemy wszystkie mozliowe znaki na tablicy, funkcja szuka wszystkich konturow i dodaje do vectora tylko te ktore moga byc znakami
        std::vector<PossibleChar> vectorOfPossibleCharsInPlate = findPossibleCharsInPlate(possiblePlate.imgGrayscale, possiblePlate.imgThresh);
        
		//funkcja znajduje grupy pasujacych znakow w tablicy
        std::vector<std::vector<PossibleChar> > vectorOfVectorsOfMatchingCharsInPlate = findVectorOfVectorsOfMatchingChars(vectorOfPossibleCharsInPlate);

        if (vectorOfVectorsOfMatchingCharsInPlate.size() == 0) {                // jesli nie znaleziono grupy pasujacych znakow
            possiblePlate.strChars = "";            //pusty string
            continue;                               //przejdz do gory petli
        }

        for (auto &vectorOfMatchingChars : vectorOfVectorsOfMatchingCharsInPlate) {                                         //dla kazdego wektora pasujacych znakow w aktualnej tablicy
            std::sort(vectorOfMatchingChars.begin(), vectorOfMatchingChars.end(), PossibleChar::sortCharsLeftToRight);      //sortuj znaki od lewej do prawej
            vectorOfMatchingChars = removeInnerOverlappingChars(vectorOfMatchingChars);                                     //usuwanie nakladajacych sie znakow
        }

		//w kazdej mozliwej tablicy, przypuszczamy ze najdluzszy wektor pasujacych znakow jest prawidlowym wektorem znakow
        unsigned int intLenOfLongestVectorOfChars = 0;
        unsigned int intIndexOfLongestVectorOfChars = 0;
        
		//szukanie najdluzszego wektora pasujacych znakow, pobranie jego indeksu
        for (unsigned int i = 0; i < vectorOfVectorsOfMatchingCharsInPlate.size(); i++) {
            if (vectorOfVectorsOfMatchingCharsInPlate[i].size() > intLenOfLongestVectorOfChars) {
                intLenOfLongestVectorOfChars = vectorOfVectorsOfMatchingCharsInPlate[i].size();
                intIndexOfLongestVectorOfChars = i;
            }
        }

		//przypuszczamy, ze najdluzszy wektor pasujacych znakow w tablicy jest prawidlowym wektorem znakow
        std::vector<PossibleChar> longestVectorOfMatchingCharsInPlate = vectorOfVectorsOfMatchingCharsInPlate[intIndexOfLongestVectorOfChars];

		//rozpoznajemy znaki na najdluzszym wektorze pasujacych znakow w tablicy
        possiblePlate.strChars = recognizeCharsInPlate(possiblePlate.imgThresh, longestVectorOfMatchingCharsInPlate);

    }

    return(vectorOfPossiblePlates); //zwracamy vectorOfPossiblePlates
}

std::vector<PossibleChar> findPossibleCharsInPlate(cv::Mat &imgGrayscale, cv::Mat &imgThresh) {
    std::vector<PossibleChar> vectorOfPossibleChars;                            //to bedziemy zwracac

    cv::Mat imgThreshCopy;

    std::vector<std::vector<cv::Point> > contours;

    imgThreshCopy = imgThresh.clone();				//wymagana kopia zeby nie zmodyfikowac

    cv::findContours(imgThreshCopy, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);        //szukamy wszystkich konturow na tablicy

    for (auto &contour : contours) {                            //dla kazdego konturu
        PossibleChar possibleChar(contour);

        if (checkIfPossibleChar(possibleChar)) {                //jesli kontur jest mozliwym znakiem
            vectorOfPossibleChars.push_back(possibleChar);      //dodaj do vectorOfPossibleChars
        }
    }

    return(vectorOfPossibleChars); //zwracamy vectorOfPossibleChars
}

bool checkIfPossibleChar(PossibleChar &possibleChar) {
    
	//funkcja sprawdzajaca, czy kontur moze byc znakiem (bez porownywania)
    if (possibleChar.boundingRect.area() > MIN_PIXEL_AREA &&
        possibleChar.boundingRect.width > MIN_PIXEL_WIDTH && possibleChar.boundingRect.height > MIN_PIXEL_HEIGHT &&
        MIN_ASPECT_RATIO < possibleChar.dblAspectRatio && possibleChar.dblAspectRatio < MAX_ASPECT_RATIO) {
        return(true);
    }
    else {
        return(false);
    }
}

std::vector<std::vector<PossibleChar> > findVectorOfVectorsOfMatchingChars(const std::vector<PossibleChar> &vectorOfPossibleChars) {

	//zaczynamy z wszystkimi mozliwymi znakami w jednym duzym wektorze, dzielimy wektor na wektor wektorow pasujacych znakow
    std::vector<std::vector<PossibleChar> > vectorOfVectorsOfMatchingChars;             //to zwracamy

    for (auto &possibleChar : vectorOfPossibleChars) {                  //dla kazdego mozliwego znaku w duzym wektorze

                                                                        //szukaj w duzym wektorze znakow ktore pasuja do aktualnego znaku
        std::vector<PossibleChar> vectorOfMatchingChars = findVectorOfMatchingChars(possibleChar, vectorOfPossibleChars);

        vectorOfMatchingChars.push_back(possibleChar);          //dodaj aktualny znak do aktualnego vectorOfMatchingChars

                                                                // jesli aktualny vectorOfMatchingChars nie jest wystarczajaco dlugi, aby mogl byc tablica (min 3 znaki)
        if (vectorOfMatchingChars.size() < MIN_NUMBER_OF_MATCHING_CHARS) {
            continue;                       // sprobuj z nastepnym znakiem
                                           
        }
        
		//tutaj mamy juz grupe pasujacych znakow (minimum 3)
        vectorOfVectorsOfMatchingChars.push_back(vectorOfMatchingChars);            //dodajemy je do vectorOfVectorsOfMatchingChars
                                                                                    
		//usuwamy aktualny wektor pasujacych znakow z duzego wektora, zeby nie uzywac tych samych znakow 2 razy
        std::vector<PossibleChar> vectorOfPossibleCharsWithCurrentMatchesRemoved;

        for (auto &possChar : vectorOfPossibleChars) {
            if (std::find(vectorOfMatchingChars.begin(), vectorOfMatchingChars.end(), possChar) == vectorOfMatchingChars.end()) {
                vectorOfPossibleCharsWithCurrentMatchesRemoved.push_back(possChar);
            }
        }
        //nowy wektor znakow
        std::vector<std::vector<PossibleChar> > recursiveVectorOfVectorsOfMatchingChars;

        //rekurencja
        recursiveVectorOfVectorsOfMatchingChars = findVectorOfVectorsOfMatchingChars(vectorOfPossibleCharsWithCurrentMatchesRemoved);

        for (auto &recursiveVectorOfMatchingChars : recursiveVectorOfVectorsOfMatchingChars) {      //dla kazdego wektora pasujacych znakow
            vectorOfVectorsOfMatchingChars.push_back(recursiveVectorOfMatchingChars);               //dodaj do oryginalnego vectorOfVectorsOfMatchingChars
        }

        break;		//wyjdz z petli
    }

    return(vectorOfVectorsOfMatchingChars); //zwracamy vectorOfVectorsOfMatchingChars
}

std::vector<PossibleChar> findVectorOfMatchingChars(const PossibleChar &possibleChar, const std::vector<PossibleChar> &vectorOfChars) {
    
	//dajemy funkcji possibleChar i vectorOfChars, szukamy wszystkich znakow w duzym wektorze, ktore pasuja do jednego possibleChar
	//i zwracamy pasujace znaki jako wektor
    std::vector<PossibleChar> vectorOfMatchingChars;                //to zwrocimy

    for (auto &possibleMatchingChar : vectorOfChars) {              //dla kazdego znaku w duzym wektorze

		//jesli znak do ktorego szukamy pasujacych znakow jest taki sam jak znak z duzego wektora ktory aktualnie sprawdzamy
        if (possibleMatchingChar == possibleChar) {          
			//nie dodajemy go do wektora, bo mielibysmy dwa te same znaki(zdublowalibysmy aktualny znak)
            continue;           //wracamy do poczatku petli
        }
        // obliczenia czy znaki pasuja
		//pasujace znaki to takie, ktore sa blisko siebie
        double dblDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar);
        double dblAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar);
        double dblChangeInArea = (double)abs(possibleMatchingChar.boundingRect.area() - possibleChar.boundingRect.area()) / (double)possibleChar.boundingRect.area();
        double dblChangeInWidth = (double)abs(possibleMatchingChar.boundingRect.width - possibleChar.boundingRect.width) / (double)possibleChar.boundingRect.width;
        double dblChangeInHeight = (double)abs(possibleMatchingChar.boundingRect.height - possibleChar.boundingRect.height) / (double)possibleChar.boundingRect.height;

        //sprawdzamy czy znaki pasuja
        if (dblDistanceBetweenChars < (possibleChar.dblDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) &&
            dblAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS && //mniej niz 12 stopni
            dblChangeInArea < MAX_CHANGE_IN_AREA && //mniejsze niz 50%
            dblChangeInWidth < MAX_CHANGE_IN_WIDTH && //mniejsze niz 80%
            dblChangeInHeight < MAX_CHANGE_IN_HEIGHT) // mniejsze niz 20% 
			{
            vectorOfMatchingChars.push_back(possibleMatchingChar);      //jesli znaki pasuja dodaj aktualny znak do vectorOfMatchingChars
        }
    }

    return(vectorOfMatchingChars);          //zwracamy vectorOfMatchingChars
}

//obliczanie dystansu miedzy znakami
double distanceBetweenChars(const PossibleChar &firstChar, const PossibleChar &secondChar) {
    int intX = abs(firstChar.intCenterX - secondChar.intCenterX);
    int intY = abs(firstChar.intCenterY - secondChar.intCenterY);

    return(sqrt(pow(intX, 2) + pow(intY, 2)));
}

// obliczanie k¹ta pomiedzy znakami
double angleBetweenChars(const PossibleChar &firstChar, const PossibleChar &secondChar) {
    double dblAdj = abs(firstChar.intCenterX - secondChar.intCenterX);
    double dblOpp = abs(firstChar.intCenterY - secondChar.intCenterY);

    double dblAngleInRad = atan(dblOpp / dblAdj);

    double dblAngleInDeg = dblAngleInRad * (180.0 / CV_PI);

    return(dblAngleInDeg);
}

//funkcja ktora usuwa nakladajace sie znaki, jesli sa za blisko siebie lub nachodza na siebie, to usuwa wewnetrzny znak (mniejszy)
//np litera "O" moze byc wykryta 2 razy (zewnetrzny i wewnetrzny kontur), wiec bierzemy tylko zewnetrzny, aby nie dublowac
std::vector<PossibleChar> removeInnerOverlappingChars(std::vector<PossibleChar> &vectorOfMatchingChars) {
    std::vector<PossibleChar> vectorOfMatchingCharsWithInnerCharRemoved(vectorOfMatchingChars);

    for (auto &currentChar : vectorOfMatchingChars) {
        for (auto &otherChar : vectorOfMatchingChars) {
            if (currentChar != otherChar) {                         // if current char and other char are not the same char . . .
                                                                    // if current char and other char have center points at almost the same location . . .
                if (distanceBetweenChars(currentChar, otherChar) < (currentChar.dblDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY)) {
                    // if we get in here we have found overlapping chars
                    // next we identify which char is smaller, then if that char was not already removed on a previous pass, remove it

                    // if current char is smaller than other char
                    if (currentChar.boundingRect.area() < otherChar.boundingRect.area()) {
                        // look for char in vector with an iterator
                        std::vector<PossibleChar>::iterator currentCharIterator = std::find(vectorOfMatchingCharsWithInnerCharRemoved.begin(), vectorOfMatchingCharsWithInnerCharRemoved.end(), currentChar);
                        // if iterator did not get to end, then the char was found in the vector
                        if (currentCharIterator != vectorOfMatchingCharsWithInnerCharRemoved.end()) {
                            vectorOfMatchingCharsWithInnerCharRemoved.erase(currentCharIterator);       // so remove the char
                        }
                    }
                    else {        // else if other char is smaller than current char
                                  // look for char in vector with an iterator
                        std::vector<PossibleChar>::iterator otherCharIterator = std::find(vectorOfMatchingCharsWithInnerCharRemoved.begin(), vectorOfMatchingCharsWithInnerCharRemoved.end(), otherChar);
                        // if iterator did not get to end, then the char was found in the vector
                        if (otherCharIterator != vectorOfMatchingCharsWithInnerCharRemoved.end()) {
                            vectorOfMatchingCharsWithInnerCharRemoved.erase(otherCharIterator);         // so remove the char
                        }
                    }
                }
            }
        }
    }

    return(vectorOfMatchingCharsWithInnerCharRemoved);
}


// tu dokonujemy rozpoznawania znakow
std::string recognizeCharsInPlate(cv::Mat &imgThresh, std::vector<PossibleChar> &vectorOfMatchingChars) {
    std::string strChars;               //bedziemy zwracac znaki z tablicy

    cv::Mat imgThreshColor;

    //sortuj znaki od lewej do prawej
    std::sort(vectorOfMatchingChars.begin(), vectorOfMatchingChars.end(), PossibleChar::sortCharsLeftToRight);

	//kolorowa wersja threshold image, by moc rysowac kontury w kolorze
    cv::cvtColor(imgThresh, imgThreshColor, CV_GRAY2BGR);       

    for (auto &currentChar : vectorOfMatchingChars) {           //dla kazdego znaku w tablicy
        cv::rectangle(imgThreshColor, currentChar.boundingRect, SCALAR_GREEN, 2);       // rysuj zielony box dookola znaku

        cv::Mat imgROItoBeCloned = imgThresh(currentChar.boundingRect);                 // pobierz ROI image boxa

        cv::Mat imgROI = imgROItoBeCloned.clone();      // kopia, zeby nie modyfikowac oryginalu

        cv::Mat imgROIResized;
        //zmiana rozmiaru obrazu, KONIECZNE DO ROZPOZNAWANIA ZNAKOW
        cv::resize(imgROI, imgROIResized, cv::Size(RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT));

        cv::Mat matROIFloat;
		//konwersja MAT do float, wymagane zeby uzywac findNearest
        imgROIResized.convertTo(matROIFloat, CV_32FC1);         

        cv::Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1);

        cv::Mat matCurrentChar(0, 0, CV_32F);                   


		//uzywamy findNearest
        kNearest->findNearest(matROIFlattenedFloat, 1, matCurrentChar);     

        float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);       

        strChars = strChars + char(int(fltCurrentChar));        //dopisujemy aktualny znak do stringa
    }

    return(strChars);               //zwracamy nasz string
}

