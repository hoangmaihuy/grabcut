#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#define DEBUG 1
#ifdef DEBUG
#  define D(x) (x)
#else
#  define D(x) do{}while(0)
#endif

enum EditMode {
    DEFAULT,
    DRAW_RECT,
    ADD_BGD_SEED,
    ADD_FGD_SEED
};

const Scalar RED = Scalar(0, 0, 255);
const Scalar BLUE = Scalar(255, 0, 0);
const Scalar GREEN = Scalar(0, 255, 0);

class GrabCut
{
    enum { BGD, FGD, UKN };
public:
    static const int radius = 2;
    static const int thickness = -1;
    int width, height, nCluster, iterCount;
    Rect rect;
    EditMode mode;
    bool isRectInit, inProgress;
    Mat mask;
    Mat img;
    vector<Point> fgdSeeds, bgdSeeds;
    GrabCut(const Mat &img, int nCluster, int iterCount) : nCluster(nCluster), iterCount(iterCount)
    {
        mode = DEFAULT;
        width = img.cols;
        height = img.rows;
        img.copyTo(this->img);
        isRectInit = false;
        fgdSeeds.clear();
        bgdSeeds.clear();
        inProgress = false;
    }

    void showInput()
    {
        Mat res;
        this->img.copyTo(res);
        for (auto it : fgdSeeds)
            circle(res, it, radius, BLUE, thickness);
        for (auto it : bgdSeeds)
            circle(res, it, radius, RED, thickness);
        if (isRectInit)
            rectangle(res, Point(rect.x, rect.y), Point(rect.x + rect.width, rect.y + rect.height), GREEN, 2);
        imshow("Input", res);
    }

    void showOutput()
    {
        Mat res;
        this->img.copyTo(res);
        imshow("Output", res);
    }

    void setMode(EditMode mode) 
    {
        this->mode = mode;
    }

    void addBGDSeed(Point p) { bgdSeeds.push_back(p); }
    void addFGDSeed(Point p) { fgdSeeds.push_back(p); }

    void clear()
    {
        isRectInit = false;
        fgdSeeds.clear();
        bgdSeeds.clear();
        inProgress = false;
        showInput();
    }

    void nextIter() 
    {

    }

    void on_mouse(int event, int x, int y, int flags, void* param);

};

string inPath, outPath;
int nCluster;
int iterCount;
GrabCut* gc;

void GrabCut::on_mouse(int event, int x, int y, int flags, void* param)
{
    if (mode == DEFAULT) return;
    if (event == EVENT_MOUSEMOVE && !inProgress) return;
    if (event == EVENT_LBUTTONDOWN)
    {
        D(cerr << "Left mouse down\n");
        inProgress = true;
        if (mode == DRAW_RECT)
        {
            rect = Rect(x, y, 1, 1);
            isRectInit = true;
        }
        else if (mode == ADD_BGD_SEED)
            addBGDSeed(Point(x, y));
        else if (mode == ADD_FGD_SEED)
            addFGDSeed(Point(x, y));
        showInput();
    }
    else if (event == EVENT_MOUSEMOVE)
    {
        if (mode == DRAW_RECT)
            rect = Rect(Point(rect.x, rect.y), Point(x, y));
        else if (mode == ADD_BGD_SEED)
            addBGDSeed(Point(x, y));
        else if (mode == ADD_FGD_SEED)
            addFGDSeed(Point(x, y));
        showInput();
    }
    else if (event == EVENT_LBUTTONUP)
    {
        D(cerr << "Left mouse up\n");
        if (mode == DRAW_RECT)
        {
            rect = Rect(Point(rect.x, rect.y), Point(x, y));
            D(cerr << rect);
        }
        else if (mode == ADD_BGD_SEED)
            addBGDSeed(Point(x, y));
        else if (gc->mode == ADD_FGD_SEED)
            addFGDSeed(Point(x, y));
        inProgress = false;
        showInput();
    }
}

void on_mouse(int event, int x, int y, int flags, void* param)
{
    gc->on_mouse(event, x, y, flags, param);
}

int main(int argc, char* argv[])
{
	const string keys = 
        "{help h usage ? |      | print this message   }"
        "{@input         |lena.jpg| input image   }"
        "{clusters k     | 5    | GMMs cluster number }"
        "{output o       |result.jpg| output image   }"
        "{count c        | 0    | interation counts, set 0 to run until converge }"
        ;
	CommandLineParser parser(argc, argv, keys);
    if (parser.has("help")) 
    {
        parser.printMessage();
        return 0;
    }

    inPath = parser.get<string>(0);
    outPath = parser.get<string>("output");
    nCluster = parser.get<int>("clusters");
    iterCount = parser.get<int>("count");
    Mat img = imread(inPath, IMREAD_COLOR);
    imshow("Input", img);
    setMouseCallback("Input", on_mouse, 0);
    imshow("Output", img);
    gc =  new GrabCut(img, nCluster, iterCount);
    while (true) 
    {
        char key = (char)waitKey(0);
        switch (key)
        {
        case '\x1b':
            D(cerr << "mode: DEFAULT\n");
            gc->setMode(DEFAULT);
            break;
        case 'b':
            D(cerr << "mode: ADD_B_SEED\n");
            gc->setMode(ADD_BGD_SEED);
            break;
        case 'f':
            D(cerr << "mode: ADD_F_SEED\n");
            gc->setMode(ADD_FGD_SEED);
            break;
        case 'r':
            D(cerr << "mode: DRAW_RECT\n");
            gc->setMode(DRAW_RECT);
            break;
        case 'n':
            gc->nextIter();
        case 'q':
            cout << "Exiting...";
            destroyAllWindows();
            return 0;
        case 'c':
            D(cerr << "Clear input\n");
            gc->clear();
        }
    }
}