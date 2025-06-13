import tensorflow as tf

# 加载原始模型
model = tf.keras.models.load_model('m.keras')

# 保存为 SavedModel 格式，移除优化器状态
model.save('model_no_optimizer', include_optimizer=False)