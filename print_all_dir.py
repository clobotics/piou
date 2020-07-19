import os
def get_dirlist(dir, Dirlist):
  newDir = dir
  if os.path.isfile(dir):
    pass
    #Dirlist.append(dir)
  elif os.path.isdir(dir):
    Dirlist.append(dir)
    for s in os.listdir(dir):
      newDir = os.path.join(dir, s)    
      get_dirlist(newDir, Dirlist)

  return Dirlist

if __name__ == '__main__':
  list = get_dirlist(os.getcwd(), [])  
  print(len(list))  
  for e in list:
    print(e)
    command = 'touch {}/__init__.py'.format(e)
    os.system(command)
