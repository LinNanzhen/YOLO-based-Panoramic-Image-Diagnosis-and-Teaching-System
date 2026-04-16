import colorsys
import numpy as np
if not hasattr(np.concatenate, '__wrapped__'):          
    _old_concat = np.concatenate
    np.concatenate = lambda arrs, axis=0, out=None, **kw: _old_concat(arrs, axis=axis, out=out)
import os
import sys
from pathlib import Path
from datetime import datetime
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json

class DataValidator:
    """数据目录与标签格式验证"""
    
    def __init__(self, data_root, label_root, num_classes=3):
        self.data_root = Path(data_root)
        self.label_root = Path(label_root)
        self.num_classes = num_classes
        self.trainset_dir = self.data_root / "trainset"
        self.testset_dir = self.data_root / "testset"
    
    def validate(self):
        """执行全面验证"""
        print("=" * 60)
        print("数据验证中...")
        print("=" * 60)
        
        # 检查目录存在性
        if not self.trainset_dir.exists():
            raise FileNotFoundError(f"❌ 训练集目录不存在: {self.trainset_dir}")
        if not self.testset_dir.exists():
            raise FileNotFoundError(f"❌ 测试集目录不存在: {self.testset_dir}")
        if not self.label_root.exists():
            raise FileNotFoundError(f"❌ 标签目录不存在: {self.label_root}")
        
        # 获取支持的图像格式
        image_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        train_images = [f for f in self.trainset_dir.iterdir() 
                       if f.suffix.lower() in image_formats]
        test_images = [f for f in self.testset_dir.iterdir() 
                      if f.suffix.lower() in image_formats]
        
        print(f"✓ 训练集: {len(train_images)} 张图像")
        print(f"✓ 测试集: {len(test_images)} 张图像")
        
        # 验证标签对应关系
        missing_labels = []
        for img in train_images:
            label_file = self.label_root / f"{img.stem}.txt"
            if not label_file.exists():
                missing_labels.append(img.name)
        
        if missing_labels:
            raise FileNotFoundError(
                f"❌ {len(missing_labels)} 张图像缺少标签:\n" +
                "\n".join(missing_labels[:5]) +
                (f"\n... 和 {len(missing_labels)-5} 张其他" 
                 if len(missing_labels) > 5 else "")
            )
        
        print(f"✓ 所有图像都有对应标签")
        
        # 抽样验证标签格式
        sample_labels = list(self.label_root.glob("*.txt"))[:3]
        for label_file in sample_labels:
            self._validate_label_format(label_file, self.num_classes)
        
        print(f"✓ 标签格式检查通过 (抽样 {len(sample_labels)} 个文件)")
        print("=" * 60)
        print()
        
        return len(train_images), len(test_images)
    
    @staticmethod
    def _validate_label_format(label_file, num_classes):
        """验证单个标签文件格式"""
        with open(label_file, 'r') as f:
            lines = f.readlines()

        for line_idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                raise ValueError(
                    f"❌ {label_file.name} 第 {line_idx+1} 行格式错误: "
                    f"期望5列，得到{len(parts)}列"
                )

            try:
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]

                if not 0 <= class_id < num_classes:
                    raise ValueError(f"类别ID {class_id} 超出范围 [0,{num_classes-1}]")

                for coord in coords:
                    if not 0 <= coord <= 1:
                        raise ValueError(f"坐标值 {coord} 超出范围 [0,1]")
            except ValueError as e:
                raise ValueError(f"❌ {label_file.name} 格式错误: {e}")


class DentalYOLOPipeline:
    """完整的训练与评估流程"""

    @staticmethod
    def _generate_colors_bgr(n):
        """根据类别数量自动生成 BGR 颜色列表（HSV 均匀采样）"""
        colors = {}
        for i in range(n):
            h = i / max(n, 1)
            r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.90)
            colors[i] = (int(b * 255), int(g * 255), int(r * 255))  # BGR
        return colors

    def __init__(self, data_root, label_root, output_dir, pretrained_weights,
                 model_size='nano', class_names=None):
        """
        初始化流程
        
        Args:
            data_root: 图像根目录
            label_root: 标签根目录
            output_dir: 输出目录
            model_size: nano/small/medium
        """
        self.data_root = Path(data_root)
        self.label_root = Path(label_root)
        self.output_dir = Path(output_dir)
        self.trainset_dir = self.data_root / "trainset"
        self.testset_dir = self.data_root / "testset"
        self.model_size = model_size
        self.pretrained_weights = pretrained_weights

        # 类别名称与颜色
        if class_names is None:
            class_names = ['Caries', 'Restoration', 'Impacted tooth']
        self.CLASS_NAMES = {i: name for i, name in enumerate(class_names)}
        self.CLASS_COLORS = DentalYOLOPipeline._generate_colors_bgr(len(class_names))
        self.num_classes = len(class_names)

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_subdir = self.output_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_subdir.mkdir(parents=True, exist_ok=True)

        # 验证数据
        validator = DataValidator(data_root, label_root, self.num_classes)
        self.n_train, self.n_test = validator.validate()
        
        # 初始化模型
        self.model = None
        self.training_log = {
            'start_time': datetime.now().isoformat(),
            'config': {
                'model_size': model_size,
                'data_root': str(data_root),
                'n_train_images': self.n_train,
                'n_test_images': self.n_test
            },
            'metrics': []
        }
        
        print(f"✓ 流程初始化完成")
        print(f"  输出目录: {self.results_subdir}")
        print()
    
    def train(self, epochs=50, batch_size=8, patience=10, lr0=0.001, device=0):
        """
        训练YOLOv8模型
        
        Args:
            epochs: 训练轮数
            batch_size: 批大小
            patience: 早停耐心值（不使用时设为-1）
            lr0: 初始学习率
            device: GPU设备ID (-1为CPU)
        """
        print("=" * 60)
        print(f"YOLOv8{self.model_size.upper()} 训练开始")
        print("=" * 60)
        
        # 创建YAML格式数据配置
        yaml_content = self._create_dataset_yaml()
        yaml_path = self.results_subdir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"✓ 数据配置文件已创建: {yaml_path}\n")
        
        if self.pretrained_weights and Path(self.pretrained_weights).exists():
            print(f"✓ 使用指定的本地预训练权重: {self.pretrained_weights}")
            self.model = YOLO(self.pretrained_weights)
        else:
            # 如果没有提供路径或路径无效，则回退到默认行为
            model_name = f"yolov8{self.model_size}.pt"
            print(f"⚠️ 未指定或未找到本地权重，将使用默认模型: {model_name} (可能需要下载)")
            self.model = YOLO(model_name)
        
        # 训练
        print(f"{'Epoch':<8} {'Loss':<12} {'Class':<12} {'Images':<10} {'Instances':<12}")
        print("-" * 60)
        
        results = self.model.train(
            data=str(yaml_path),
            epochs=epochs,
            batch=batch_size,
            patience=patience if patience > 0 else 999,  # 高值禁用早停
            lr0=lr0,
            device=device if device >= 0 else 'cpu',
            verbose=False,
            save=True,
            project=str(self.output_dir),
            name=self.results_subdir.name,
            exist_ok=True
        )
        
        self.training_log['end_time'] = datetime.now().isoformat()
        self.training_log['final_metrics'] = {
            'best_epoch': results.epoch if hasattr(results, 'epoch') else 'N/A',
            'best_fitness': results.fitness if hasattr(results, 'fitness') else 'N/A'
        }
        
        print("\n" + "=" * 60)
        print("训练完成！")
        print("=" * 60)
        print(f"✓ 模型已保存: {self.results_subdir}/weights")
        print()
        
        # 保存训练日志
        self._save_training_log()
    
    def evaluate_and_visualize(self, confidence_threshold=0.5, 
                               max_visualize=None, save_json=True):
        """
        评估测试集并可视化结果
        
        Args:
            confidence_threshold: 置信度阈值 [0, 1]
            max_visualize: 最多可视化多少张图像 (None=全部)
            save_json: 是否保存JSON格式检测结果
        """
        if self.model is None:
            # 加载最佳模型
            best_model_path = self.results_subdir / 'weights' / 'best.pt'
            if not best_model_path.exists():
                raise FileNotFoundError(f"模型文件不存在: {best_model_path}")
            self.model = YOLO(str(best_model_path))
        
        print("=" * 60)
        print("测试集评估与可视化")
        print("=" * 60)
        print(f"置信度阈值: {confidence_threshold}")
        print()
        
        test_images = sorted([f for f in self.testset_dir.iterdir() 
                             if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])
        
        if max_visualize:
            test_images = test_images[:max_visualize]
        
        vis_results = []
        detections_json = []
        
        for idx, img_path in enumerate(test_images, 1):
            print(f"[{idx}/{len(test_images)}] 处理: {img_path.name}")
            
            # 推理
            results = self.model.predict(source=str(img_path), conf=confidence_threshold, 
                                        verbose=False)
            result = results[0]
            
            # 读取原始图像
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"  ❌ 无法读取图像")
                continue
            
            h, w = image.shape[:2]
            
            # 提取检测结果
            detections = []
            for box, cls, conf in zip(result.boxes.xywh, result.boxes.cls, result.boxes.conf):
                cx, cy, bw, bh = box[:4].cpu().numpy()
                class_id = int(cls.cpu().numpy())
                confidence = float(conf.cpu().numpy())
                
                # 转换为像素坐标
                x1 = max(0, int(cx - bw/2))
                y1 = max(0, int(cy - bh/2))
                x2 = min(w, int(cx + bw/2))
                y2 = min(h, int(cy + bh/2))
                
                detections.append({
                    'class_id': class_id,
                    'class_name': self.CLASS_NAMES[class_id],
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2]
                })
                
                # 绘制检测框
                color = self.CLASS_COLORS[class_id]
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # 标签文本
                label = f"{self.CLASS_NAMES[class_id]}: {confidence:.2f}"
                cv2.putText(image, label, (x1, max(20, y1-5)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 保存可视化结果
            output_path = self.results_subdir / 'visualizations' / img_path.name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), image)
            
            print(f"  ✓ 检测到 {len(detections)} 个对象，已保存")
            
            vis_results.append({
                'image_name': img_path.name,
                'detections': detections,
                'output_path': str(output_path)
            })
            
            detections_json.append({
                'image_name': img_path.name,
                'image_size': [w, h],
                'detections': detections
            })
        
        # 保存JSON结果
        if save_json:
            json_path = self.results_subdir / 'detections.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(detections_json, f, indent=2, ensure_ascii=False)
            print(f"\n✓ 检测结果已保存: {json_path}")
        
        print("=" * 60)
        print(f"✓ 可视化完成！结果保存在: {self.results_subdir}/visualizations")
        print("=" * 60)
        print()
    
    def _create_dataset_yaml(self):
        """创建YAML格式的数据集配置"""
        names_lines = "\n".join(
            f"  {i}: {name}" for i, name in self.CLASS_NAMES.items()
        )
        yaml_content = f"""
path: {self.data_root.resolve()}
train: trainset
val: trainset
test: testset

nc: {self.num_classes}
names:
{names_lines}
"""
        return yaml_content.strip()
    
    def _save_training_log(self):
        """保存训练日志"""
        log_path = self.results_subdir / 'training_log.json'
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_log, f, indent=2, ensure_ascii=False)


def main():
    """主程序入口"""
    print("\n" + "=" * 60)
    print("YOLOv8 口腔X线影像诊断系统 - 训练模块")
    print("=" * 60)
    print()
    
    # ==================== 配置参数 ====================
    # [学生可修改区域] 开始
    
    # 1. 文件路径配置
    data_root = input("请输入图像根目录路径 (默认: ./images): ").strip() or "./images"
    label_root = input("请输入标签根目录路径 (默认: ./labels/trainset): ").strip() or "./labels/trainset"
    output_dir = input("请输入输出目录路径 (默认: ./results): ").strip() or "./results"
    
    # 2. 预训练权重配置
    pretrained_weights = input("请输入本地预训练权重文件路径 (例如: ./yolov8n.pt, 留空则使用默认模型): ").strip()
    if not pretrained_weights:
        pretrained_weights = "./yolov8n.pt" # 如果用户直接回车，则设为 None
        
    # 3. 模型配置
    print("\n模型尺寸选择:")
    print("  1: nano   (快速，推荐用于教学)")
    print("  2: small  (均衡)")
    print("  3: medium (准度高，但较慢)")
    model_choice = input("选择 (默认: 1): ").strip() or "1"
    model_sizes = {"1": "nano", "2": "small", "3": "medium"}
    model_size = model_sizes.get(model_choice, "nano")
    
    # 4. 训练参数
    epochs_str = input("训练轮数 (默认: 50): ").strip() or "50"
    epochs = max(1, int(epochs_str))
    
    batch_size_str = input("批大小 (默认: 8): ").strip() or "8"
    batch_size = max(1, int(batch_size_str))
    
    patience_str = input("早停耐心值 (-1=禁用，默认: 10): ").strip() or "10"
    patience = int(patience_str)
    
    lr_str = input("初始学习率 (默认: 0.001): ").strip() or "0.001"
    lr0 = float(lr_str)
    
    # 5. 可视化参数
    confidence_str = input("检测置信度阈值 (0-1，默认: 0.5): ").strip() or "0.5"
    confidence_threshold = float(confidence_str)
    
    device_str = input("GPU设备ID (-1=CPU，默认: 0): ").strip() or "0"
    device = int(device_str)
    
    # [学生可修改区域] 结束
    
    print("\n" + "=" * 60)
    print("配置总结:")
    print("=" * 60)
    print(f"数据根目录: {data_root}")
    print(f"标签根目录: {label_root}")
    print(f"输出目录: {output_dir}")
    print(f"模型: YOLOv8{model_size}")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}")
    print(f"初始学习率: {lr0}, Early Stop Patience: {patience}")
    print(f"可视化置信度: {confidence_threshold}")
    print(f"计算设备: {'GPU' if device >= 0 else 'CPU'}")
    print("=" * 60)
    
    proceed = input("\n确认开始训练? (y/n，默认: y): ").strip().lower() or "y"
    if proceed != 'y':
        print("已取消")
        return
    
    # ==================== 执行流程 ====================
    
    try:
        # 1. 初始化流程
        pipeline = DentalYOLOPipeline(
            data_root=data_root,
            label_root=label_root,
            output_dir=output_dir,
            model_size=model_size,
            pretrained_weights=pretrained_weights
        )
        
        # 2. 开始训练
        pipeline.train(
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            lr0=lr0,
            device=device
        )
        
        # 3. 评估与可视化
        print("\n开始测试集评估...")
        pipeline.evaluate_and_visualize(
            confidence_threshold=confidence_threshold,
            save_json=True
        )
        
        print("\n" + "=" * 60)
        print("✓ 所有步骤完成!")
        print(f"✓ 结果保存在: {pipeline.results_subdir}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 错误发生: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()