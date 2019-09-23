from musket_core import image_datasets,datasets

class BinaryTextClassificationDataSet(image_datasets.BinaryClassificationDataSet):
    
    def __init__(self,csvPath,textColumn,clazzColumn):
        super().__init__(None, csvPath, textColumn, clazzColumn)
    
    def _id(self,item):
        return item
    
    def _encode_x(self, item):
        return self[item.id].x    
        
    def get_value(self,t):
        return t
    
class CategoryTextClassificationDataSet(image_datasets.CategoryClassificationDataSet):
    
    def __init__(self,csvPath,textColumn,clazzColumn):
        super().__init__(None, csvPath, textColumn, clazzColumn)
    
    def _encode_x(self, item):
        return self[int(item.id)].x    
        
    def get_value(self,t):
        return t        

class MultiClassTextClassificationDataSet(image_datasets.MultiClassClassificationDataSet):
    
    def __init__(self,csvPath,textColumn,clazzColumn):
        super().__init__(None, csvPath, textColumn, clazzColumn)
    
    def _encode_x(self, item):
        return self[int(item.id)].x    
        
    def get_value(self,t):
        return t    