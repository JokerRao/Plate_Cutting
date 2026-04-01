export type ProjectCache = {
  name: string;
  cuttingPlans: any[];
  orders: any[];
  others: any[];
};

export const _projectCache = new Map<string, ProjectCache>();

export function invalidateLayoutCache(projectId?: string) {
  if (projectId) {
    _projectCache.delete(projectId);
  } else {
    _projectCache.clear();
  }
}
