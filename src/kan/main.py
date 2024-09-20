import os
import json
import torch
import argparse

from torch.nn import CrossEntropyLoss
from torchmetrics.classification import MulticlassAccuracy as Accuracy

from train import trainer
from distil import distiller
from dataset import get_mnist
from configs import update_config, cfg
from plots import plot_curves, plot_curve


train_path = "../trainings"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str)
    args = parser.parse_args()
    update_config(cfg, args)
    
    cfg_file = args.cfg[args.cfg.find("configs/")+len("configs/"):args.cfg.find(".yaml")]
    train_path = os.path.join(train_path, cfg_file)
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    
    train_loader, test_loader = get_mnist(cfg.AUGMENTATION)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if cfg.TYPE == 'mlp':
        from nets import make_mlp_student as make_student
    elif cfg.TYPE == 'kan':
        from nets import make_kan_student as make_student
    if cfg.TEACHER_TYPE == 'mlp':
        from nets import make_mlp_teacher as make_teacher
    elif cfg.TEACHER_TYPE == 'kan':
        from nets import make_kan_teacher as make_teacher

    teacher, teacher_sched, teacher_optimiz = make_teacher(device, cfg)
    student, student_sched, student_optimiz = make_student(device, cfg.TRAINER)
    
    # if cfg.TRAINED_TEACHER is not None:
    #     teacher.load_state_dict(torch.load(cfg.TRAINED_TEACHER))
    #     with open(cfg.TRAINED_TEACHER[:cfg.TRAINED_TEACHER.find(".pth")]+"_metr.json", 'r') as j:
    #         teacher_metrics = json.loads(j.read())
    #     with open(cfg.TRAINED_TEACHER.replace("teacher", "student")[:cfg.TRAINED_TEACHER.find(".pth")]+"_metr.json", 'r') as j:
    #         student_metrics = json.loads(j.read())
        
    #     model, metrics = distiller(student, teacher, device, train_loader, test_loader, student_optimiz, student_sched, cfg)
    #     torch.save(model.state_dict(), os.path.join(train_path, f'{cfg_file}_stud_dist.pth'))
    #     with open(os.path.join(train_path,f'{cfg_file}_stud_dist_metr.json'), "w") as outfile: 
    #         json.dump(metrics, outfile)
    #     plot_curves(model, teacher, metrics, teacher_metrics, train_path, cfg_file, 'teacher')
    #     plot_curves(model, model, metrics, student_metrics, train_path, cfg_file, 'student')
    
    # else:
    if cfg.MODEL == 'student':
        model = student
        optimiz = student_optimiz
        sched = student_optimiz
    else:
        model = teacher
        optimiz = teacher_optimiz
        sched = teacher_optimiz

    if cfg.TRAINER == 'custom':
        model, metrics = trainer(model, device, train_loader, test_loader, optimiz, sched, cfg)
    elif cfg.TRAINER == 'orig':
        metrics = model.train(train_loader, test_loader, cfg=cfg, lamb=0., loss_fn=CrossEntropyLoss(),
                                metrics=Accuracy(num_classes=10).to(device))
    elif cfg.TRAINER == 'distil':
        print("loading student from", cfg.TRAINED_TEACHER.replace("teacher", "student")[:cfg.TRAINED_TEACHER.find(".pth")]+"_metr.json")
        print("loading teacher from", cfg.TRAINED_TEACHER)
        teacher.load_state_dict(torch.load(os.path.join(cfg.TRAINED_TEACHER, "teacher.pth")))
        if cfg.TEACHER_TYPE == 'mlp':
            teacher.eval()
        if cfg.STUDENT_METRICS != '':
            with open(os.path.join(cfg.STUDENT_METRICS, "student_metr.json"), 'r') as j:
                student_metrics = json.loads(j.read())
        else:
            # with open(cfg.TRAINED_TEACHER.replace("teacher", "student")[:cfg.TRAINED_TEACHER.find(".pth")]+"_metr.json", 'r') as j:
            with open(os.path.join(cfg.TRAINED_TEACHER.replace("teacher", "student"), "student_metr.json"), 'r') as j:
                student_metrics = json.loads(j.read())
        with open(os.path.join(cfg.TRAINED_TEACHER, "teacher_metr.json"), 'r') as j:
            teacher_metrics = json.loads(j.read())
        metrics = model.distil(teacher, train_loader, test_loader, cfg=cfg, lamb=0.,
                                loss_fn=CrossEntropyLoss(), metrics=Accuracy(num_classes=10).to(device))
        plot_curves(model, model, metrics, student_metrics, train_path, cfg_file, 'student')
        plot_curves(model, teacher, metrics, teacher_metrics, train_path, cfg_file, 'teacher')

    print(metrics)
    torch.save(model.state_dict(), os.path.join(train_path, f'{cfg.MODEL}.pth'))
    with open(os.path.join(train_path,f'{cfg.MODEL}_metr.json'), "w") as outfile: 
        json.dump(metrics, outfile)
    plot_curve(metrics, train_path, cfg_file)
    # model.plot(folder=train_path)
