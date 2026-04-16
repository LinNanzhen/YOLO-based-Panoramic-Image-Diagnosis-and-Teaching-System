import numpy as np
if not hasattr(np.concatenate, '__wrapped__'):          
    _old_concat = np.concatenate
    np.concatenate = lambda arrs, axis=0, out=None, **kw: _old_concat(arrs, axis=axis, out=out)
import sys
from pathlib import Path
from datetime import datetime
import cv2
from ultralytics import YOLO
import json


class DentalYOLOVisualizer:
    """YOLOv8可视化器"""
    
    CLASS_NAMES = {0: 'Caries', 1: 'Restoration', 2: 'Impacted tooth'}
    CLASS_COLORS = {0: (0, 0, 255), 1: (0, 255, 0), 2: (255, 0, 0)}  # BGR
    
    def __init__(self, model_path, test_dir, output_dir=None):
        """
        初始化可视化器
        
        Args:
            model_path: 训练好的模型权重路径 (.pt文件)
            test_dir: 测试图像目录
            output_dir: 输出目录 (默认为模型所在目录的visualizations文件夹)
        """
        self.model_path = Path(model_path)
        self.test_dir = Path(test_dir)
        
        # 检查路径有效性
        if not self.model_path.exists():
            raise FileNotFoundError(f"❌ 模型文件不存在: {self.model_path}")
        if not self.test_dir.exists():
            raise FileNotFoundError(f"❌ 测试目录不存在: {self.test_dir}")
        
        # 设置输出目录
        if output_dir is None:
            # 默认在模型所在目录的上两级创建visualizations目录
            self.output_dir = self.model_path.parent.parent / 'visualizations'
        else:
            self.output_dir = Path(output_dir)
        
        # 加载模型
        print("=" * 60)
        print("加载模型中...")
        print("=" * 60)
        print(f"模型路径: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        print(f"✓ 模型加载成功")
        print()
    
    def visualize(self, confidence_threshold=0.5, max_images=None, save_json=True):
        """
        可视化测试集检测结果
        
        Args:
            confidence_threshold: 置信度阈值 [0, 1]
            max_images: 最多可视化多少张图像 (None=全部)
            save_json: 是否保存JSON格式检测结果
        """
        print("=" * 60)
        print("测试集可视化")
        print("=" * 60)
        print(f"测试目录: {self.test_dir}")
        print(f"置信度阈值: {confidence_threshold}")
        print()
        
        # 创建输出目录（带时间戳）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        vis_output_dir = self.output_dir / f"conf_{confidence_threshold}_{timestamp}"
        vis_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取测试图像
        test_images = sorted([f for f in self.test_dir.iterdir() 
                             if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])
        
        if not test_images:
            print(f"❌ 在 {self.test_dir} 中未找到图像")
            return
        
        if max_images:
            test_images = test_images[:max_images]
        
        print(f"找到 {len(test_images)} 张测试图像")
        print("-" * 60)
        
        detections_json = []
        total_detections = 0
        
        for idx, img_path in enumerate(test_images, 1):
            print(f"[{idx}/{len(test_images)}] 处理: {img_path.name}", end=" ")
            
            # 先确认图像能读
            image = cv2.imread(str(img_path))
            if image is None or image.size == 0:
                print("❌ 无法读取，跳过")
                continue

            # 再预测，防止空图触发 np.stack
            try:
                results = self.model.predict(
                    source=str(img_path),
                    conf=confidence_threshold,
                    verbose=False
                )
                result = results[0]
            except ValueError as e:
                if "need at least one array to stack" in str(e):
                    print("❌ 图像空/损坏，跳过")
                    continue
                raise
            
            # 读取原始图像
            image = cv2.imread(str(img_path))
            if image is None:
                print("❌ 无法读取")
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
            output_path = vis_output_dir / img_path.name
            cv2.imwrite(str(output_path), image)
            
            total_detections += len(detections)
            print(f"✓ 检测到 {len(detections)} 个对象")
            
            detections_json.append({
                'image_name': img_path.name,
                'image_size': [w, h],
                'detections': detections
            })
        
        # 保存JSON结果
        if save_json:
            json_path = vis_output_dir / 'detections.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(detections_json, f, indent=2, ensure_ascii=False)
            print(f"\n✓ 检测结果已保存: {json_path}")
        
        # 统计信息
        print("\n" + "=" * 60)
        print("可视化完成!")
        print("=" * 60)
        print(f"✓ 处理图像数: {len(test_images)}")
        print(f"✓ 总检测对象: {total_detections}")
        print(f"✓ 平均每张: {total_detections/len(test_images):.1f} 个对象")
        print(f"✓ 结果保存在: {vis_output_dir}")
        print("=" * 60)
        print()


def main():
    """主程序入口"""
    print("\n" + "=" * 60)
    print("YOLOv8 口腔X线影像诊断系统 - 可视化模块")
    print("=" * 60)
    print()
    
    # ==================== 配置参数 ====================
    
    # 1. 模型路径
    print("请输入训练好的模型权重路径")
    print("示例: ./results/run_20241019_143022/weights/best.pt")
    model_path = input("模型路径: ").strip()
    
    if not model_path:
        print("❌ 必须提供模型路径")
        sys.exit(1)
    
    # 2. 测试图像目录
    test_dir = input("请输入测试图像目录 (默认: ./images/testset): ").strip() or "./images/testset"
    
    # 3. 输出目录（可选）
    output_dir = input("请输入可视化输出目录 (留空则使用默认位置): ").strip() or "./images/vis"
    
    # 4. 置信度阈值
    while True:
        confidence_str = input("检测置信度阈值 (0-1，默认: 0.5): ").strip() or "0.5"
        try:
            confidence_threshold = float(confidence_str)
            if 0 <= confidence_threshold <= 1:
                break
            else:
                print("❌ 置信度必须在 0-1 之间")
        except ValueError:
            print("❌ 请输入有效的数字")
    
    # 5. 最大图像数
    max_images_str = input("最多可视化多少张图像 (留空=全部): ").strip()
    max_images = int(max_images_str) if max_images_str else None
    
    # 6. 是否保存JSON
    save_json_str = input("是否保存JSON检测结果? (y/n，默认: y): ").strip().lower() or "y"
    save_json = save_json_str == 'y'
    
    print("\n" + "=" * 60)
    print("配置总结:")
    print("=" * 60)
    print(f"模型路径: {model_path}")
    print(f"测试目录: {test_dir}")
    print(f"输出目录: {output_dir or '(自动)'}")
    print(f"置信度阈值: {confidence_threshold}")
    print(f"最多图像数: {max_images or '全部'}")
    print(f"保存JSON: {save_json}")
    print("=" * 60)
    
    proceed = input("\n确认开始可视化? (y/n，默认: y): ").strip().lower() or "y"
    if proceed != 'y':
        print("已取消")
        return
    
    # ==================== 执行可视化 ====================
    
    try:
        visualizer = DentalYOLOVisualizer(
            model_path=model_path,
            test_dir=test_dir,
            output_dir=output_dir
        )
        
        visualizer.visualize(
            confidence_threshold=confidence_threshold,
            max_images=max_images,
            save_json=save_json
        )
        
    except Exception as e:
        print(f"\n❌ 错误发生: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()