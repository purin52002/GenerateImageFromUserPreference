from GUI.image_canvas_frame import ImageCanvas
from tkinter import Tk, LEFT, Frame, Label, LabelFrame

from ImageEnhancer.image_enhancer import ImageEnhancer

from UserPreferencePredictor.Model.util \
    import MODEL_TYPE_LIST

from UserPreferencePredictor.Model.Compare.ranknet import RankNet
from ScoredParamIO.scored_param_reader import read_scored_param

from argparse import ArgumentParser


class EvaluatedCanvasFrame(Frame):
    def __init__(self, master: Frame, canvas_width: int,
                 canvas_height: int, score: float, evaluate: float):
        super(EvaluatedCanvasFrame, self).__init__(master)
        self.canvas = ImageCanvas(self, canvas_width, canvas_height)

        self.canvas.pack()
        Label(self, text='score: %f' % (score)).pack()
        Label(self, text='evaluate: %.2f' % (evaluate)).pack()


def _get_args():
    parser = ArgumentParser()

    parser.add_argument('-l', '--load_dir_path', required=True)
    parser.add_argument('-i', '--image_path', required=True)
    parser.add_argument('-p', '--param_file_path', required=True)
    parser.add_argument('-t', '--model_type', choices=MODEL_TYPE_LIST,
                        required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = _get_args()

    for arg in vars(args):
        print(f'{str(arg)}: {str(getattr(args, arg))}')

    model_type = args.model_type

    predict_model = RankNet()

    try:
        predict_model.load(args.load_dir_path)
    except ValueError:
        print('ロードができなかったため終了します')
        exit()

    image_enhancer = ImageEnhancer(args.image_path)

    scored_param_list = read_scored_param(args.param_file_path)

    data_list = \
        [{'image': image_enhancer.org_enhance(scored_param),
            'score': scored_param['score']}
         for scored_param in scored_param_list]

    evaluate_list = predict_model.predict(data_list)

    for i in range(len(data_list)):
        data_list[i]['evaluate'] = evaluate_list[i][0]

    high_predicted_data_list = \
        sorted(data_list, key=lambda x: x['evaluate'], reverse=True)

    high_scored_data_list = \
        sorted(data_list, key=lambda x: x['score'], reverse=True)

    root = Tk()
    root.attributes('-topmost', True)

    disp_num = 4

    target_width = 300
    target_height = \
        int(target_width/image_enhancer.org_width*image_enhancer.org_height)

    high_predict_frame = LabelFrame(root, text='predict')
    for predicted_data in high_predicted_data_list[:disp_num]:
        frame = \
            EvaluatedCanvasFrame(high_predict_frame,
                                 target_width,
                                 target_height,
                                 predicted_data['score'],
                                 predicted_data['evaluate'])
        frame.pack(side=LEFT)
        frame.canvas.update_image(predicted_data['image'])
    high_predict_frame.pack(pady=10)

    high_score_frame = LabelFrame(root, text='score')
    for scored_data in high_scored_data_list[:disp_num]:
        frame = \
            EvaluatedCanvasFrame(high_score_frame,
                                 target_width,
                                 target_height,
                                 scored_data['score'], scored_data['evaluate'])
        frame.pack(side=LEFT)
        frame.canvas.update_image(scored_data['image'])
    high_score_frame.pack(pady=10)

    root.mainloop()
