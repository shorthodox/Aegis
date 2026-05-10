export class AuthManager {
    static getToken() {
        return localStorage.getItem('access_token') || sessionStorage.getItem('access_token');
    }

    static setToken(token, remember = true) {
        if (remember) {
            localStorage.setItem('access_token', token);
        } else {
            sessionStorage.setItem('access_token', token);
        }
    }
    
    static clearToken() {
        localStorage.removeItem('access_token');
        sessionStorage.removeItem('access_token');
    }
    
    static isAuthenticated() {
        return !!this.getToken();
    }
    
    static getAuthHeader() {
        const token = this.getToken();
        return token ? `Bearer ${token}` : null;
    }

    static setUser(userData) {
        localStorage.setItem('user_profile', JSON.stringify(userData));
        if (userData) {
            let endDate = null;
            if (userData.trial && userData.trial.endDate) {
                endDate = userData.trial.endDate;
            } else if (userData.trial_end) {
                endDate = userData.trial_end;
            } else if (userData.trialEnd) {
                endDate = userData.trialEnd;
            }
            
            if (endDate) {
                // Support both Firestore Timestamp and ISO String
                if (typeof endDate === 'object' && endDate.seconds) {
                    endDate = new Date(endDate.seconds * 1000).toISOString();
                } else if (typeof endDate === 'number') { // Support ms timestamp
                    endDate = new Date(endDate).toISOString();
                }
                localStorage.setItem('trial_end_timestamp', endDate);
                console.log('[AuthManager] trial_end_timestamp updated:', endDate);
            } else {
                console.warn('[AuthManager] No trial end date found in userData:', userData);
            }
        }
    }

    static getUser() {
        const data = localStorage.getItem('user_profile');
        return data ? JSON.parse(data) : null;
    }

    static getPlanType() {
        const tokenData = this.getUserData();
        if (tokenData && tokenData.plan_type) {
            return tokenData.plan_type;
        }
        
        const user = this.getUser();
        if (user && user.plan) {
            return user.plan;
        }
        
        return 'free_trial';
    }

    static isTrialValid() {
        const trialEnd = localStorage.getItem('trial_end_timestamp');
        if (!trialEnd) return false;
        return new Date() < new Date(trialEnd);
    }

    static isTokenValid() {
        const token = this.getToken();
        if (!token) return false;
        try {
            const payload = JSON.parse(atob(token.split('.')[1]));
            // Add a small 10 second buffer for token expiration instead of 5 minutes
            return payload.exp * 1000 > Date.now() + 10000;
        } catch (e) {
            return false;
        }
    }

    static async refreshToken() {
        try {
            const response = await fetch('/api/auth/refresh', {
                method: 'POST',
                headers: {
                    'Authorization': this.getAuthHeader()
                }
            });
            if (response.ok) {
                const data = await response.json();
                if (data.access_token) {
                    this.setToken(data.access_token);
                    return true;
                }
            }
            return false;
        } catch (error) {
            console.error('Token refresh failed', error);
            return false;
        }
    }

    static getUserData() {
        const token = this.getToken();
        if (!token) return null;
        try {
            const payload = JSON.parse(atob(token.split('.')[1]));
            return {
                trial_start: payload.trial_start,
                plan_type: payload.plan_type,
                status: payload.status
            };
        } catch (e) {
            console.error('Failed to decode token payload', e);
            return null;
        }
    }

    static hasAccess(feature) {
        const data = this.getUserData();
        if (!data) return false;
        
        let hasBaseAccess = false;
        if (data.plan_type === 'active' || data.plan_type === 'pro' || data.status === 'active') {
            hasBaseAccess = true;
        } else if (data.plan_type === 'free_trial' && data.status !== 'expired') {
            // Trust the status/plan_type from token instead of hardcoded 24-hour limit
            hasBaseAccess = true;
        }
        
        if (!hasBaseAccess) return false;
        
        if (feature === 'signals') return true;
        if (feature === 'extended_timeframes') {
            return data.plan_type === 'active' || data.plan_type === 'pro';
        }
        
        return false;
    }

    static getSubscriptionStatus() {
        // Prioritize decoded JWT as single source of truth
        const tokenData = this.getUserData();
        if (tokenData) {
            return this.hasAccess('signals') ? 'active' : 'expired';
        }
        
        // Fallbacks
        const user = this.getUser();
        if (user && (user.plan === 'pro' || user.plan === 'active')) {
            return 'active';
        }
        
        if (this.isTrialValid()) {
            return 'active';
        }
        
        return 'expired';
    }
}
