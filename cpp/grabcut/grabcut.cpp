#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "GMM.h"
#include "graph.h"

using namespace cv;
using namespace std;

typedef Graph<double, double, double> GraphType;

#define DEBUG 1
#ifdef DEBUG
#  define D(x) (x)
#else
#  define D(x) do{}while(0)
#endif

#define REP(i, n) for (int i = 0; i < n; i++)
#define For(i, a, b) for (int i = a; i <= b; i++)

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
    enum Matte { BGD, FGD, PR_BGD, PR_FGD};
public:
    static const int radius = 2;
    static const int thickness = -1;
    static const int neighbor_num = 8;
    const int dx[4] = { 1, 0, -1, 1 };
    const int dy[4] = { 0, 1, 1, 1 };
    const double dis[4] = { 1, 1, 1.0 / sqrt(2), 1.0/sqrt(2) };
    GraphType* g;
    int width, height, nCluster, iterCount, edgeNum, nodeNum;
    const double gamma = 50.0, maxEdge = gamma * 8;
    double beta;
    Rect rect;
    EditMode mode;
    bool isRectInit, inProgress, initMask;
    int iter;
    Matte** mask;
    Mat img;
    vector<Vec3b> bgdPixs, fgdPixs;
    vector<pair<int, int>> bgdPos, fgdPos;
    VecIndex bgdComp, fgdComp;
    vector<Point> fgdSeeds, bgdSeeds;
    GMM bgdModel, fgdModel;

    GrabCut(const Mat &img, int nCluster, int iterCount) : nCluster(nCluster), iterCount(iterCount)
    {
        mode = DEFAULT;
        width = img.cols;
        height = img.rows;
        D(cerr << "(h, w) = " << height << " " << width << "\n");
        img.copyTo(this->img);
        createMask();
        clear();
    }

    void createMask()
    {
        mask = new Matte* [height];
        REP(y, height)
        {
            mask[y] = new Matte[width];
        }
        D(cerr << "mask created!\n");
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
        REP(y, height) REP(x, width)
        {
            if (mask[y][x] == BGD || mask[y][x] == PR_BGD)
                res.at<Vec3b>(y, x) = Vec3b(0, 0, 0);
        }
        imshow("Output", res);
    }

    void setMode(EditMode mode) 
    {
        this->mode = mode;
    }

    void addBGDSeed(Point p) { bgdSeeds.push_back(p); mask[p.y][p.x] = BGD; }
    void addFGDSeed(Point p) { fgdSeeds.push_back(p); mask[p.y][p.x] = FGD; }

    void clearPixs()
    {
        fgdPixs.clear(); bgdPixs.clear();
        fgdPos.clear(); bgdPos.clear();
        bgdComp.clear(); fgdComp.clear();
    }

    void clear()
    {
        clearPixs();
        REP(y, height) REP(x, width) mask[y][x] = BGD;
        initMask = false;
        isRectInit = false;
        fgdSeeds.clear(); bgdSeeds.clear();
        inProgress = false;
        iter = 0;
    }

    void initIter()
    {
        calcBeta();
        int x1 = max(rect.x, 0), y1 = max(rect.y, 0);
        int x2 = min(x1 + rect.width, width-1), y2 = min(y1 + rect.height, height-1);
        D(cerr << "rect = " << x1 << " " << y1 << " " << x2 << " " << y2 << "\n");
        For(y, y1, y2) For(x, x1, x2)
            mask[y][x] = PR_FGD;
        rebuildPixs();
        bgdModel.init_components(bgdPixs, bgdComp);
        fgdModel.init_components(fgdPixs, fgdComp);
    }

    void nextIter() 
    {
        D(cerr << "iter: " << iter << "\n");
        if (!iter) initIter();
        if (initMask) rebuildPixs();
        assignGMM(bgdPixs, bgdComp, bgdModel);
        assignGMM(fgdPixs, fgdComp, fgdModel);
        bgdModel.learn(bgdPixs, bgdComp);
        fgdModel.learn(fgdPixs, fgdComp);
        buildGraph();
        graphCut();
        showOutput();
        iter++;
    }

    void assignGMM(const vector<Vec3b>& pixels, VecIndex &components, GMM &model)
    {
        REP(i, pixels.size())
            components[i] = model.get_component(pixels[i]);
    }

    void calcBeta()
    {
        double dist = 0.0;
        edgeNum = 0;
        REP(y, height) REP(x, width)
        {
            REP(k, neighbor_num / 2)
            {
                int u = x + dx[k], v = y + dy[k];
                if (u >= 0 && v >= 0 && u < width && v < height)
                {
                    //D(cerr << x << ' ' << y << ' ' << u << ' ' << v << "\n");
                    auto diff = img.at<Vec3b>(y, x) - img.at<Vec3b>(v, u);
                    REP(t, 3) dist += (double)diff[t] * diff[t];
                    edgeNum++;
                }
            }
        }
        beta = 0.5 / (dist / edgeNum);
        D(cerr << "beta = " << beta << "\n");
    }

    inline int to1DCoord(int x, int y) { return y * width + x; }

    void buildGraph()
    {
        D(cerr << "Building graph...\n");
        nodeNum = height * width;
        edgeNum += 2 * nodeNum;
        g = new GraphType(nodeNum, edgeNum);
        REP(i, nodeNum) g->add_node();
        REP(y, height) REP(x, width)
        {
            int s = to1DCoord(x, y);
            auto pixel = img.at<Vec3b>(y, x);
            // calc T-links
            double bgd_w = 0, fgd_w = 0;
            if (mask[y][x] == BGD)
            {
                bgd_w = maxEdge, fgd_w = 0;
            }
            else if (mask[y][x] == FGD)
            {
                bgd_w = 0; fgd_w = maxEdge;
            }
            else
            {
                /* our version */
                bgd_w = -log(fgdModel.model_likelihood(pixel)); 
                fgd_w = -log(bgdModel.model_likelihood(pixel));
                /* paper version */
                //bgd_w = fgdModel.model_likelihood(pixel); 
                //fgd_w = bgdModel.model_likelihood(pixel);
                //D(cerr << "t_weights = " << bgd_w << " " << fgd_w << "\n");
            }
            g->add_tweights(s, bgd_w, fgd_w);

            // calc N-links     
            REP(k, neighbor_num/2)
            {
                int u = x + dx[k], v = y + dy[k];
                if (u >= 0 && v >= 0 && u < width && v < height)
                {
                    int t = to1DCoord(u, v);
                    auto diff = pixel - img.at<Vec3b>(v, u);
                    double mult = 0;
                    REP(t, 3) mult += (double)diff[t] * diff[t];
                    //with inverse distance
                    double w = gamma * dis[k] * exp(-beta * mult);
                    // without inverse distance
                    // double w = gamma * exp(-beta * mult);
                    g->add_edge(s, t, w, w);
                    //D(cerr << "n_weights = " << w << "\n");
                }
            }
        }
        D(cerr << "Done building graph\n");
    }

    inline void addBGDPix(int x, int y)
    {
        bgdPixs.push_back(img.at<Vec3b>(y, x));
        bgdPos.push_back(make_pair(x, y));
        bgdComp.push_back(0);
    }

    inline void addFGDPix(int x, int y)
    {
        fgdPixs.push_back(img.at<Vec3b>(y, x));
        fgdPos.push_back(make_pair(x, y));
        fgdComp.push_back(0);
    }

    void graphCut()
    {
        D(cerr << "Doing graph cut...\n");
        double flow = g->maxflow();
        
        REP(i, nodeNum)
        {
            int y = i / width, x = i % width;
            if (mask[y][x] != BGD && mask[y][x] != FGD)
                if (g->what_segment(i) == GraphType::SOURCE)
                    mask[y][x] = PR_BGD;
                else
                    mask[y][x] = PR_FGD;
            
        }
        D(cerr << "Done graph cut\n");
    }

    void rebuildPixs()
    {
        clearPixs();
        for (auto p : bgdSeeds) mask[p.y][p.x] = BGD;
        for (auto p : fgdSeeds) mask[p.y][p.x] = FGD;
        REP(y, height) REP(x, width)
            if (mask[y][x] == BGD || mask[y][x] == PR_BGD)
                addBGDPix(x, y);
            else
                addFGDPix(x, y);

    }

    void saveResult(string outPath)
    {
        cerr << outPath << "\n";
        Mat res;
        this->img.copyTo(res);
        REP(y, height) REP(x, width)
        {
            if (mask[y][x] == BGD || mask[y][x] == PR_BGD)
                res.at<Vec3b>(y, x) = Vec3b(0, 0, 0);
        }
        imwrite(outPath, res);
        D(cerr << "result saved to " + outPath << "\n");
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
            D(cerr << rect << "\n");
        }
        else if (mode == ADD_BGD_SEED)
        {
            addBGDSeed(Point(x, y)); initMask = true;
        }
        else if (gc->mode == ADD_FGD_SEED)
        {
            addFGDSeed(Point(x, y)); initMask = true;
        }
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
        "{clusters     | 5    | GMMs cluster number }"
        "{output       |result.jpg| output image   }"
        "{count        | 1    | interation counts}"
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
    D(cerr << "inPath = " + inPath << "\n");
    D(cerr << "outPath = " + outPath << "\n");
    D(cerr << "nCluster = " << nCluster << "\n");
    D(cerr << "iterCount = " << iterCount << "\n");
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
            break;
        case 'q':
            cout << "Exiting...";
            destroyAllWindows();
            return 0;
        case 'c':
            D(cerr << "Clear input\n");
            gc->clear();
            gc->showInput();
            break;
        case 's':
            gc->saveResult(outPath);
            break;
        }
    }
}